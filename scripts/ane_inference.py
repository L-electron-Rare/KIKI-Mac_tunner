#!/usr/bin/env python3
"""Inference ANE optimisee — import direct d'ANEMLL, pas de subprocess.

Gain: 8x vs subprocess (6.5s->0.8s par appel) en eliminant le rechargement du modele.
Supporte le scoring parallele (3 instances) et le batch prefill.

Utilise directement les fonctions de /tmp/anemll/tests/chat.py :
- load_model() : charge un .mlmodelc ou .mlpackage via coremltools
- load_metadata() : extrait les parametres du modele (context_length, batch_size, etc.)
- initialize_tokenizer() : charge le tokenizer HuggingFace
- run_prefill() : prefill batch via ANE (batch_size=64 par defaut)
- generate_next_token() : decode un token a la fois via ANE
- create_unified_state() : cree le KV cache partage entre les chunks
"""

import sys
import os
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import torch
import yaml

# Ajouter ANEMLL au path pour importer chat.py
ANEMLL_PATH = os.environ.get("ANEMLL_PATH", "/tmp/anemll")
sys.path.insert(0, os.path.join(ANEMLL_PATH, "tests"))

# Import des fonctions internes d'ANEMLL (tests/chat.py)
from chat import (
    load_model,
    load_metadata,
    load_models,
    initialize_tokenizer,
    initialize_causal_mask,
    create_unified_state,
    run_prefill,
    generate_next_token,
    build_stop_token_ids,
    parse_model_path,
    parse_ffn_filename,
    find_all_chunks,
    detect_cache_type,
    make_causal_mask,
)


class _FakeArgs:
    """Simule les args d'argparse pour les fonctions ANEMLL qui attendent args."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _build_args_from_meta(meta_path: str, context_length: Optional[int] = None):
    """Construit un objet args a partir d'un fichier meta.yaml."""
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)

    params = meta["model_info"]["parameters"]
    model_dir = str(Path(meta_path).parent)
    model_type = meta["model_info"].get("model_type", "chunked")
    is_monolithic = model_type == "monolithic"

    prefix = params.get("model_prefix", "qwen")
    lut_bits = params.get("lut_bits", "none")
    lut_ffn = params.get("lut_ffn", lut_bits)
    lut_lmhead = params.get("lut_lmhead", lut_bits)
    lut_embeddings = params.get("lut_embeddings", lut_bits)

    ctx = context_length or int(params["context_length"])

    args = _FakeArgs(
        d=model_dir,
        meta=meta_path,
        embed=None,
        ffn=None,
        pf=None,
        lmhead=None,
        tokenizer=params.get("tokenizer_path", model_dir),
        context_length=ctx,
        batch_size=int(params.get("batch_size", 64)),
        num_chunks=int(params.get("num_chunks", 1)),
        num_logits=int(params.get("num_logits", 8)),
        split_lm_head=int(params.get("split_lm_head", 16 if "qwen" in prefix.lower() else 8)),
        sliding_window=params.get("sliding_window"),
        attention_size=int(params.get("attention_size", ctx)),
        cpu=False,
        eval=True,  # Mode silencieux
        nw=True,
        no_template=False,
        no_think=False,
        debug=False,
        debug_argmax=False,
        mem_report=False,
        split_rotate=bool(params.get("split_rotate", False)),
        is_monolithic=is_monolithic,
        argmax_in_model=params.get("argmax_in_model", False),
        vocab_size=params.get("vocab_size"),
        lm_head_chunk_sizes=params.get("lm_head_chunk_sizes"),
        update_mask_prefill=params.get("update_mask_prefill", False),
        prefill_dynamic_slice=params.get("prefill_dynamic_slice", False),
    )

    if is_monolithic:
        lut_suffix = f"_lut{lut_bits}" if lut_bits != "none" else ""
        args.monolithic_model = params.get(
            "monolithic_model", f"{prefix}_monolithic_full{lut_suffix}.mlmodelc"
        )
        args.state_length = int(params.get("state_length", ctx))
    else:
        # Construire les chemins des modeles chunks
        def _lut_suffix(bits):
            return f"_lut{bits}" if bits != "none" else ""

        num_c = int(params.get("num_chunks", 1))
        if not args.embed:
            args.embed = params.get("embeddings") or f"{prefix}_embeddings{_lut_suffix(lut_embeddings)}"
            args.embed = args.embed.replace(".mlmodelc", "").replace(".mlpackage", "")
        if not args.lmhead:
            args.lmhead = params.get("lm_head") or f"{prefix}_lm_head{_lut_suffix(lut_lmhead)}"
            args.lmhead = args.lmhead.replace(".mlmodelc", "").replace(".mlpackage", "")
        if not args.ffn:
            ffn_candidate = params.get("ffn")
            if ffn_candidate:
                args.ffn = ffn_candidate.replace(".mlmodelc", "").replace(".mlpackage", "")
            else:
                args.ffn = f"{prefix}_FFN_PF{_lut_suffix(lut_ffn)}_chunk_01of{num_c:02d}"
        if args.split_rotate and not args.pf:
            pf_candidate = params.get("pf")
            if pf_candidate:
                args.pf = pf_candidate.replace(".mlmodelc", "").replace(".mlpackage", "")
            else:
                args.pf = f"{prefix}_PF{_lut_suffix(lut_ffn)}_chunk_01of{num_c:02d}"

    return args, params


class ANEModel:
    """Modele ANE charge en memoire — reutilisable sans rechargement.

    Charge les modeles CoreML (embed, FFN chunks, LM head) une seule fois,
    puis genere du texte via prefill batch + decode token par token.
    """

    def __init__(self, model_dir: str, context_length: Optional[int] = None):
        """Charge le modele CoreML ANE une seule fois.

        Args:
            model_dir: Dossier contenant meta.yaml et les .mlmodelc/.mlpackage
            context_length: Longueur de contexte (defaut: depuis meta.yaml)
        """
        meta_path = Path(model_dir) / "meta.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"meta.yaml introuvable dans {model_dir}. "
                "Convertir le modele avec ANEMLL d'abord."
            )

        self._args, self._params = _build_args_from_meta(str(meta_path), context_length)
        self._model_dir = Path(model_dir)
        self._is_monolithic = self._args.is_monolithic

        # Charger le tokenizer
        tokenizer_path = str(Path(self._args.tokenizer).resolve())
        self.tokenizer = initialize_tokenizer(tokenizer_path, eval_mode=True)

        # Charger les modeles et creer le state
        if self._is_monolithic:
            self._load_monolithic()
        else:
            self._load_chunked()

        # Construire les stop tokens
        self._stop_token_ids = build_stop_token_ids(self.tokenizer)

        # Detecter le support du chat template
        self._has_chat_template = False
        try:
            test_msgs = [{"role": "user", "content": "test"}]
            self.tokenizer.apply_chat_template(test_msgs, tokenize=True, return_dict=False)
            self._has_chat_template = True
        except Exception:
            pass

    def _load_chunked(self):
        """Charge les modeles en mode chunks (embed + FFN chunks + LM head)."""
        args = self._args
        model_dir = Path(args.d).resolve()

        # Rendre les chemins absolus
        args.embed = str(model_dir / args.embed)
        args.ffn = str(model_dir / args.ffn)
        args.lmhead = str(model_dir / args.lmhead)
        if args.split_rotate and args.pf:
            args.pf = str(model_dir / args.pf)

        metadata = {}
        self.embed_model, self.ffn_models, self.lmhead_model, metadata = load_models(args, metadata)

        # Enrichir metadata — types explicites, match chat.py main() lignes 2554-2575
        metadata["context_length"] = int(args.context_length)
        metadata["state_length"] = int(args.context_length)
        metadata["batch_size"] = int(metadata.get("batch_size", 64))
        metadata["num_logits"] = int(getattr(args, "num_logits", 8))
        metadata["split_lm_head"] = int(getattr(args, "split_lm_head", 16))
        metadata["debug"] = False
        metadata["argmax_in_model"] = getattr(args, "argmax_in_model", False)
        metadata["debug_argmax"] = False
        metadata["vocab_size"] = getattr(args, "vocab_size", None)
        metadata["sliding_window"] = getattr(args, "sliding_window", None)
        metadata["attention_size"] = int(getattr(args, "attention_size", args.context_length))
        metadata["update_mask_prefill"] = getattr(args, "update_mask_prefill", False)
        metadata["prefill_dynamic_slice"] = getattr(args, "prefill_dynamic_slice", False)
        # lm_head_chunk_sizes — manquant dans la v1, requis par generate_next_token
        try:
            from chat import _parse_lm_head_chunk_sizes
            metadata["lm_head_chunk_sizes"] = _parse_lm_head_chunk_sizes(
                getattr(args, "lm_head_chunk_sizes", None)
            )
        except ImportError:
            metadata["lm_head_chunk_sizes"] = None
        self.metadata = metadata

        # KV cache state
        self.state = create_unified_state(
            self.ffn_models, metadata["context_length"], eval_mode=True, metadata=metadata
        )

        # Masque causal
        attention_size = metadata.get("attention_size", metadata["context_length"])
        self.causal_mask = initialize_causal_mask(attention_size, eval_mode=True)

    def _load_monolithic(self):
        """Charge le modele monolithique (un seul .mlmodelc avec toutes les fonctions)."""
        # Import de la fonction specifique au monolithique
        from chat import load_monolithic_model

        args = self._args
        metadata = {}
        (
            self._infer_model,
            self._infer_rotate_model,
            self._prefill_model,
            self._prefill_rotate_model,
            metadata,
        ) = load_monolithic_model(args, metadata)

        metadata["context_length"] = args.context_length
        metadata["state_length"] = getattr(args, "state_length", args.context_length)
        metadata["batch_size"] = getattr(args, "batch_size", 64)
        metadata["split_lm_head"] = getattr(args, "split_lm_head", 16)
        metadata["argmax_in_model"] = getattr(args, "argmax_in_model", False)
        metadata["debug_argmax"] = False
        metadata["debug"] = False
        metadata["sliding_window"] = getattr(args, "sliding_window", None)
        metadata["vocab_size"] = getattr(args, "vocab_size", None)
        self.metadata = metadata

        # State et masque causal
        self.state = self._infer_model.make_state()
        state_len = metadata.get("state_length", metadata["context_length"])
        self.causal_mask = initialize_causal_mask(state_len, eval_mode=True)

    def _reset_state(self):
        """Reinitialise le KV cache pour une nouvelle generation."""
        if self._is_monolithic:
            self.state = self._infer_model.make_state()
        else:
            self.state = create_unified_state(
                self.ffn_models, self.metadata["context_length"],
                eval_mode=True, metadata=self.metadata,
            )

    def _tokenize_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenise un prompt avec le chat template si disponible."""
        if self._has_chat_template:
            messages = [{"role": "user", "content": prompt}]
            token_ids = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=False,
            )
            input_ids = torch.tensor([token_ids], dtype=torch.int32)
        else:
            formatted = f"[INST] {prompt} [/INST]"
            token_ids = self.tokenizer.encode(formatted)
            input_ids = torch.tensor([token_ids], dtype=torch.int32)
        return input_ids

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> Optional[str]:
        """Genere du texte sur ANE sans recharger le modele.

        Args:
            prompt: Texte d'entree (question utilisateur)
            max_tokens: Nombre maximum de tokens a generer
            temperature: Temperature d'echantillonnage (0.0 = greedy)

        Returns:
            Texte genere ou None en cas d'erreur
        """
        try:
            # Reinitialiser le KV cache
            self._reset_state()

            input_ids = self._tokenize_prompt(prompt)
            context_length = self.metadata["context_length"]
            batch_size = self.metadata.get("batch_size", 64)
            context_pos = input_ids.size(1)

            if context_pos >= context_length - 1:
                return None

            # Etendre input_ids a la taille du contexte
            if input_ids.size(1) < context_length:
                padding = torch.zeros(
                    (1, context_length - input_ids.size(1)), dtype=torch.int32
                )
                input_ids = torch.cat([input_ids, padding], dim=1)

            if self._is_monolithic:
                return self._generate_monolithic(
                    input_ids, context_pos, max_tokens, temperature
                )
            return self._generate_chunked(
                input_ids, context_pos, max_tokens, temperature
            )

        except Exception as e:
            import traceback
            print(f"[ANEModel] Erreur generation: {e}")
            traceback.print_exc()
            return None

    def _generate_chunked(
        self,
        input_ids: torch.Tensor,
        context_pos: int,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generation via le pipeline chunks (embed + FFN + LM head)."""
        context_length = self.metadata["context_length"]
        batch_size = self.metadata.get("batch_size", 64)
        sliding_window = self.metadata.get("sliding_window")
        update_mask_prefill = self.metadata.get("update_mask_prefill", False)
        prefill_dynamic_slice = self.metadata.get("prefill_dynamic_slice", False)

        single_token_mode = not (update_mask_prefill or prefill_dynamic_slice)

        # Prefill
        run_prefill(
            self.embed_model,
            self.ffn_models,
            input_ids,
            context_pos,
            context_length,
            batch_size,
            self.state,
            self.causal_mask,
            sliding_window,
            single_token_mode=single_token_mode,
            use_update_mask=update_mask_prefill,
        )

        # Decode token par token
        pos = context_pos
        generated_ids = []

        while pos < context_length - 1 and len(generated_ids) < max_tokens:
            next_token = generate_next_token(
                self.embed_model,
                self.ffn_models,
                self.lmhead_model,
                input_ids,
                pos,
                context_length,
                self.metadata,
                self.state,
                self.causal_mask,
                temperature=temperature,
            )

            input_ids[0, pos] = next_token
            generated_ids.append(next_token)
            pos += 1

            if next_token in self._stop_token_ids:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def _generate_monolithic(
        self,
        input_ids: torch.Tensor,
        context_pos: int,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """Generation via le pipeline monolithique."""
        from chat import (
            run_monolithic_prefill,
            generate_next_token_monolithic,
        )

        context_length = self.metadata["context_length"]
        batch_size = self.metadata.get("batch_size", 64)

        # Prefill monolithique
        run_monolithic_prefill(
            self._prefill_model,
            input_ids,
            context_pos,
            context_length,
            batch_size,
            self.state,
            self.causal_mask,
            infer_model=self._infer_model,
        )

        # Decode
        pos = context_pos
        generated_ids = []

        while pos < context_length - 1 and len(generated_ids) < max_tokens:
            next_token = generate_next_token_monolithic(
                self._infer_model,
                input_ids,
                pos,
                context_length,
                self.metadata,
                self.state,
                self.causal_mask,
                temperature=temperature,
            )

            input_ids[0, pos] = next_token
            generated_ids.append(next_token)
            pos += 1

            if next_token in self._stop_token_ids:
                break

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> list[Optional[str]]:
        """Batch sequentiel — genere pour plusieurs prompts l'un apres l'autre.

        Le prefill batch d'ANEMLL (batch_size=64) est deja utilise en interne
        pour amortir le cout du prefill. Chaque prompt est traite sequentiellement
        car le KV cache est partage (pas de vrai batch multi-sequence sur ANE).

        Args:
            prompts: Liste de prompts a traiter
            max_tokens: Tokens max par generation
            temperature: Temperature d'echantillonnage

        Returns:
            Liste de textes generes (None si erreur)
        """
        return [
            self.generate(prompt, max_tokens, temperature)
            for prompt in prompts
        ]


class ParallelANEScorer:
    """Instances ANE en parallele via ThreadPoolExecutor.

    CoreML supporte plusieurs modeles charges simultanement.
    Chaque worker a son propre KV cache state, donc les generations
    sont independantes.

    Note: Le gain reel depend de la bande passante ANE/memoire.
    Sur M4 avec 512 Go, 2-3 instances sont raisonnables.
    """

    def __init__(
        self,
        model_dir: str,
        num_workers: int = 3,
        context_length: Optional[int] = None,
    ):
        """Charge num_workers instances independantes du modele ANE.

        Args:
            model_dir: Dossier du modele converti (avec meta.yaml)
            num_workers: Nombre d'instances paralleles (defaut: 3)
            context_length: Override de la longueur de contexte
        """
        print(f"Chargement de {num_workers} instances ANE...")
        self.workers = []
        for i in range(num_workers):
            print(f"  Instance {i + 1}/{num_workers}...", flush=True)
            self.workers.append(ANEModel(model_dir, context_length))
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        print(f"{num_workers} instances ANE pretes.")

    def generate_parallel(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.7,
    ) -> list[Optional[str]]:
        """Genere en parallele sur plusieurs instances ANE.

        Les prompts sont repartis en round-robin sur les workers.

        Args:
            prompts: Liste de prompts
            max_tokens: Tokens max par generation
            temperature: Temperature d'echantillonnage

        Returns:
            Liste ordonnee de textes generes (None si erreur)
        """
        futures = []
        for i, prompt in enumerate(prompts):
            worker = self.workers[i % len(self.workers)]
            future = self.executor.submit(
                worker.generate, prompt, max_tokens, temperature
            )
            futures.append(future)

        results = []
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                print(f"[ParallelANEScorer] Erreur worker: {e}")
                results.append(None)
        return results

    def shutdown(self):
        """Arrete le pool de threads."""
        self.executor.shutdown(wait=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test inference ANE directe")
    parser.add_argument("--model-dir", required=True, help="Dossier du modele ANE")
    parser.add_argument("--prompt", default="What is 2+2?", help="Prompt de test")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Chargement du modele depuis {args.model_dir}...")
    t0 = time.time()
    model = ANEModel(args.model_dir)
    load_time = time.time() - t0
    print(f"Modele charge en {load_time:.1f}s")

    # Premier appel (warmup)
    print("\n--- Warmup ---")
    t0 = time.time()
    _ = model.generate("Hello", max_tokens=10, temperature=0.0)
    warmup_time = time.time() - t0
    print(f"Warmup: {warmup_time:.1f}s")

    # Deuxieme appel (temps reel)
    print(f"\n--- Generation ---")
    print(f"Prompt: {args.prompt}")
    t0 = time.time()
    result = model.generate(args.prompt, args.max_tokens, args.temperature)
    gen_time = time.time() - t0
    print(f"\nReponse: {result}")
    print(f"\nTemps: {gen_time:.1f}s")
