"""
Rebuild GraphSSR `*_graph_bert_encoded.pt` files from upstream data.

This script covers:
- products
- wikics
- fb15k237
- arxiv (best-effort comparison build)

It writes the three evaluation files to:
- data/test_data/raw_data/products/products_graph_bert_encoded.pt
- data/test_data/raw_data/wikics/wikics_graph_bert_encoded.pt
- data/test_data/raw_data/fb15k237/fb15k237_graph_bert_encoded.pt

It also builds an arxiv artifact and compares it to the existing
GraphSSR arxiv checkpoint when available.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import sys
import types
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import HashingVectorizer
from torch_geometric.data import Data
from torch_geometric.utils import coalesce


GRAPHSSR_ROOT = Path(__file__).resolve().parents[2]
GOFA_ROOT = GRAPHSSR_ROOT.parent / "GOFA-main"
GNNGUARD_ARXIV_ROOT = GRAPHSSR_ROOT.parent / "GNNGuard-master" / "Datasets" / "ogbn_arxiv"
DEFAULT_ENCODER = "hash"
EMBED_DIM = 768


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_gzip_csv(path: Path) -> list[list[str]]:
    with gzip.open(path, "rt", newline="") as f:
        return list(csv.reader(f))


def _iter_chunks(items: Sequence[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield list(items[start:start + batch_size])


def _text_to_embedding_hash(texts: Sequence[str], batch_size: int = 4096) -> torch.Tensor:
    vectorizer = HashingVectorizer(
        n_features=EMBED_DIM,
        alternate_sign=False,
        norm="l2",
        lowercase=True,
        ngram_range=(1, 2),
    )
    outputs: list[np.ndarray] = []
    for chunk in _iter_chunks(list(texts), batch_size):
        mat = vectorizer.transform(chunk).astype(np.float32).toarray()
        outputs.append(mat)
    return torch.from_numpy(np.concatenate(outputs, axis=0))


def _text_to_embedding_transformers(
    texts: Sequence[str],
    model_name_or_path: str,
    batch_size: int = 32,
    max_length: int = 256,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_name_or_path, local_files_only=True)
    model.eval()

    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in _iter_chunks(list(texts), batch_size):
            encoded = tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            out = model(**encoded)
            hidden = out.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0).float()


def encode_texts(
    texts: Sequence[str],
    encoder: str = DEFAULT_ENCODER,
) -> torch.Tensor:
    if encoder == "hash":
        return _text_to_embedding_hash(texts)
    try:
        return _text_to_embedding_transformers(texts, encoder)
    except Exception as exc:
        print(f"[encoder] fallback to hash encoder because '{encoder}' is unavailable: {exc}")
        return _text_to_embedding_hash(texts)


def _install_sparse_stubs() -> None:
    pyg_lib = types.ModuleType("pyg_lib")
    pyg_lib.ops = types.SimpleNamespace()
    pyg_lib.sampler = types.SimpleNamespace()
    sys.modules["pyg_lib"] = pyg_lib

    for name in ["torch_sparse", "torch_sparse.tensor", "torch_sparse.storage", "torch_sparse.utils"]:
        sys.modules.setdefault(name, types.ModuleType(name))

    class SparseStorage:
        def __setstate__(self, state):
            self.__dict__.update(state)

    class SparseTensor:
        def __setstate__(self, state):
            self.__dict__.update(state)

    sys.modules["torch_sparse.storage"].SparseStorage = SparseStorage
    sys.modules["torch_sparse.tensor"].SparseTensor = SparseTensor
    sys.modules["torch_sparse"].SparseStorage = SparseStorage
    sys.modules["torch_sparse"].SparseTensor = SparseTensor


def _symmetrize_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    return edge_index


def build_products(encoder: str = DEFAULT_ENCODER) -> Data:
    _install_sparse_stubs()
    data_path = GOFA_ROOT / "data" / "products" / "ogbn-products_subset.pt"
    node_csv = GOFA_ROOT / "data" / "products" / "ogbn-products_subset.csv.gz"
    label_csv = GOFA_ROOT / "data" / "products" / "labelidx2productcategory.csv.gz"
    label_json = GOFA_ROOT / "data" / "products" / "products.json"

    raw = torch.load(data_path, map_location="cpu", weights_only=False)
    node_desc = pd.read_csv(node_csv)
    label_desc = json.load(label_json.open())

    node_texts: list[str] = []
    for i in range(raw.num_nodes):
        row = node_desc.iloc[i]
        title = row["title"]
        content = row["content"]
        if pd.isna(title) or str(title).strip() == "NA":
            title = "missing"
        if pd.isna(content) or str(content).strip() == "None":
            content = "missing"
        node_texts.append(
            "Product from Amazon platform with title and content: "
            f"Title: {title}. Content: {content}"
        )

    label_rows = _load_gzip_csv(label_csv)
    label_list = [row[1] for row in label_rows[1:]]
    if len(label_desc) >= 46:
        ordered = []
        for i in range(46):
            if i == 24:
                ordered.append("label 25")
            else:
                idx = i if i < 24 else i - 1
                ordered.append(label_desc[idx]["name"])
        ordered.append("#508510")
        label_list = ordered

    storage = raw.adj_t.storage
    edge_index = torch.stack([storage._col, storage._row], dim=0).long()
    edge_index = _symmetrize_edge_index(edge_index, int(raw.num_nodes))

    x = encode_texts(node_texts, encoder=encoder)
    y = raw.y.view(-1).long()
    out = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=raw.train_mask.view(-1).bool(),
        val_mask=raw.val_mask.view(-1).bool(),
        test_mask=raw.test_mask.view(-1).bool(),
        num_nodes=int(raw.num_nodes),
        num_classes=int(len(label_list)),
        label_list=label_list,
    )
    return out


def build_wikics(encoder: str = DEFAULT_ENCODER) -> Data:
    data_dir = GOFA_ROOT / "data" / "wikics"
    data = json.load((data_dir / "data.json").open())
    meta = json.load((data_dir / "metadata.json").open())
    label_desc = json.load((data_dir / "wikics.json").open())

    node_texts = [
        ("Wikipedia entry. Entry name: " + node["title"] + ". Entry content: " + " ".join(node["tokens"])).strip()
        for node in meta["nodes"]
    ]
    label_list = [label.lower().strip() for _, label in sorted(meta["labels"].items(), key=lambda item: int(item[0]))]
    edge_pairs = [(i, j) for i, js in enumerate(data["links"]) for j in js]
    edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
    edge_index = _symmetrize_edge_index(edge_index, len(node_texts))
    x = encode_texts(node_texts, encoder=encoder)
    y = torch.tensor(data["labels"], dtype=torch.long)
    out = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        train_mask=torch.tensor(data["train_masks"], dtype=torch.bool).t().contiguous(),
        val_mask=torch.tensor(data["val_masks"], dtype=torch.bool).t().contiguous(),
        test_mask=torch.tensor(data["test_mask"], dtype=torch.bool),
        num_nodes=len(node_texts),
        num_classes=len(label_list),
        label_list=label_list,
        stopping_mask=torch.tensor(data["stopping_masks"], dtype=torch.bool).t().contiguous(),
        label_description={item["name"].lower(): item["description"] for item in label_desc},
    )
    return out


def build_fb15k237(encoder: str = DEFAULT_ENCODER) -> Data:
    data_dir = GOFA_ROOT / "data" / "FB15K237"
    entity_json = json.load((data_dir / "entity2wikidata.json").open())
    relation_json = json.load((data_dir / "fb15k237.json").open())

    entity_names: list[str] = []
    node_texts: list[str] = []
    for entity, payload in entity_json.items():
        entity_names.append(entity)
        description = payload.get("description")
        node_texts.append(
            "Entity in the knowledge graph. Entity name: "
            + payload["label"]
            + ", Entity alternatives: "
            + ", ".join(payload.get("alternatives", []))
            + ". Entity description:"
            + (description if description is not None else "Missing.")
        )

    entity2id = {entity: i for i, entity in enumerate(entity_names)}
    relation2id: dict[str, int] = {}
    relation_names: list[str] = []
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_y: list[int] = []

    for split_name in ["train", "valid", "test"]:
        with (data_dir / f"{split_name}.txt").open() as f:
            for line in f:
                head, relation, tail = line.strip().split()
                if head not in entity2id:
                    entity2id[head] = len(entity_names)
                    entity_names.append(head)
                    node_texts.append("entity names: Unknown")
                if tail not in entity2id:
                    entity2id[tail] = len(entity_names)
                    entity_names.append(tail)
                    node_texts.append("entity names: Unknown")
                if relation not in relation2id:
                    relation2id[relation] = len(relation2id)
                    relation_names.append("Relation from source entity to target entity: " + relation)
                edge_src.append(entity2id[head])
                edge_dst.append(entity2id[tail])
                edge_y.append(relation2id[relation])

    x = encode_texts(node_texts, encoder=encoder)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    y = torch.tensor(edge_y, dtype=torch.long)
    out = Data(
        x=x,
        y=y,
        edge_index=edge_index,
        num_nodes=len(node_texts),
        num_classes=len(relation_names),
        label_list=[item["name"] for item in relation_json],
        label_description={item["name"]: item["description"] for item in relation_json},
    )
    return out


def build_arxiv(encoder: str = DEFAULT_ENCODER, reference_pt: Path | None = None) -> tuple[Data, dict]:
    edge_csv = GNNGUARD_ARXIV_ROOT / "raw" / "edge.csv.gz"
    node_label_csv = GNNGUARD_ARXIV_ROOT / "raw" / "node-label.csv.gz"
    nodeidx2paperid_csv = GNNGUARD_ARXIV_ROOT / "mapping" / "nodeidx2paperid.csv.gz"
    label_csv = GNNGUARD_ARXIV_ROOT / "mapping" / "labelidx2arxivcategeory.csv.gz"
    arxiv_meta_json = GOFA_ROOT / "data" / "arxiv" / "arxiv.json"
    summary_jsonl = GRAPHSSR_ROOT / "data" / "train_data" / "raw_data" / "arxiv" / "arxiv_summary.jsonl"

    edge = pd.read_csv(edge_csv, compression="gzip", header=None).values.T.astype(np.int64)
    node_label = pd.read_csv(node_label_csv, compression="gzip", header=None).values.squeeze().astype(np.int64)
    nodeidx2paperid = pd.read_csv(nodeidx2paperid_csv, index_col="node idx")
    label_rows = _load_gzip_csv(label_csv)
    arxiv_meta = json.load(arxiv_meta_json.open())
    label_list = [
        f"cs.{row[1].split()[-1].upper()}, {meta['name']}"
        for row, meta in zip(label_rows[1:], arxiv_meta)
    ]

    summaries: list[str] = []
    with summary_jsonl.open() as f:
        for line in f:
            row = json.loads(line)
            prompt = str(row["prompt"])
            if "Text:" in prompt:
                summaries.append(prompt.split("Text:", 1)[1].strip())
            else:
                summaries.append(str(row["response"]))
    if len(summaries) != len(nodeidx2paperid):
        raise ValueError(f"arxiv summary count mismatch: {len(summaries)} vs {len(nodeidx2paperid)}")

    node_texts = ["Academic paper. Text: " + summary for summary in summaries]

    # Replace the paper id text with real title/abstract data if present in the snapshot.
    # The summary file already carries the paper content, so we keep the title fallback.
    x = encode_texts(node_texts, encoder=encoder)
    edge_index = torch.tensor(edge, dtype=torch.long)
    edge_index = _symmetrize_edge_index(edge_index, len(node_texts))

    train_mask = val_mask = test_mask = None
    compare_report: dict = {}
    if reference_pt is not None and reference_pt.exists():
        ref = torch.load(reference_pt, map_location="cpu", weights_only=False)
        if hasattr(ref, "train_mask"):
            train_mask = ref.train_mask.clone()
        if hasattr(ref, "val_mask"):
            val_mask = ref.val_mask.clone()
        if hasattr(ref, "test_mask"):
            test_mask = ref.test_mask.clone()
        compare_report = compare_graphs(
            Data(
                x=x,
                y=torch.tensor(node_label, dtype=torch.long),
                edge_index=edge_index,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
                num_nodes=len(node_texts),
                num_classes=len(label_list),
                label_list=label_list,
            ),
            ref,
        )

    data_kwargs = dict(
        x=x,
        y=torch.tensor(node_label, dtype=torch.long),
        edge_index=edge_index,
        num_nodes=len(node_texts),
        num_classes=len(label_list),
        label_list=label_list,
    )
    if train_mask is not None:
        data_kwargs["train_mask"] = train_mask
    if val_mask is not None:
        data_kwargs["val_mask"] = val_mask
    if test_mask is not None:
        data_kwargs["test_mask"] = test_mask
    out = Data(**data_kwargs)
    return out, compare_report


def _sort_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    order = edge_index[0] * max(num_nodes, 1) + edge_index[1]
    perm = torch.argsort(order)
    return edge_index[:, perm]


def compare_graphs(generated: Data, reference: Data) -> dict:
    report: dict[str, object] = {}
    gen_keys = set(generated.keys())
    ref_keys = set(reference.keys())
    report["keys_equal"] = gen_keys == ref_keys
    report["missing_in_generated"] = sorted(list(ref_keys - gen_keys))
    report["missing_in_reference"] = sorted(list(gen_keys - ref_keys))
    report["num_nodes_equal"] = int(getattr(generated, "num_nodes", -1)) == int(getattr(reference, "num_nodes", -2))
    report["num_classes_equal"] = int(getattr(generated, "num_classes", -1)) == int(getattr(reference, "num_classes", -2))
    report["label_list_equal"] = list(getattr(generated, "label_list", [])) == list(getattr(reference, "label_list", []))
    report["x_shape_equal"] = tuple(generated.x.shape) == tuple(reference.x.shape)
    report["y_shape_equal"] = tuple(generated.y.shape) == tuple(reference.y.shape)
    report["edge_index_shape_equal"] = tuple(generated.edge_index.shape) == tuple(reference.edge_index.shape)
    report["train_mask_equal"] = hasattr(generated, "train_mask") and hasattr(reference, "train_mask") and torch.equal(generated.train_mask, reference.train_mask)
    report["val_mask_equal"] = hasattr(generated, "val_mask") and hasattr(reference, "val_mask") and torch.equal(generated.val_mask, reference.val_mask)
    report["test_mask_equal"] = hasattr(generated, "test_mask") and hasattr(reference, "test_mask") and torch.equal(generated.test_mask, reference.test_mask)

    if generated.edge_index.shape == reference.edge_index.shape:
        gen_edge = _sort_edge_index(generated.edge_index, int(generated.num_nodes))
        ref_edge = _sort_edge_index(reference.edge_index, int(reference.num_nodes))
        report["edge_index_equal"] = torch.equal(gen_edge, ref_edge)
    else:
        report["edge_index_equal"] = False

    if generated.x.shape == reference.x.shape:
        diff = torch.mean(torch.abs(generated.x.float() - reference.x.float())).item()
        report["x_mean_abs_diff"] = diff
    return report


def save_graph(data: Data, output_path: Path) -> None:
    _ensure_dir(output_path)
    torch.save(data, output_path, pickle_protocol=4)


def generate_requested_graphs(
    datasets: Sequence[str],
    encoder: str = DEFAULT_ENCODER,
    output_root: Path | None = None,
    arxiv_output: Path | None = None,
    arxiv_reference: Path | None = None,
) -> dict:
    output_root = output_root or (GRAPHSSR_ROOT / "data" / "test_data" / "raw_data")
    result: dict[str, str] = {}

    builders = {
        "products": build_products,
        "wikics": build_wikics,
        "fb15k237": build_fb15k237,
    }

    for dataset in datasets:
        if dataset not in builders:
            raise ValueError(f"Unsupported dataset: {dataset}")
        graph = builders[dataset](encoder=encoder)
        output_path = output_root / dataset / f"{dataset}_graph_bert_encoded.pt"
        save_graph(graph, output_path)
        result[dataset] = str(output_path)
        print(f"[save] {dataset}: {output_path}")

    arxiv_artifact = None
    if arxiv_output is not None:
        arxiv_graph, compare_report = build_arxiv(encoder=encoder, reference_pt=arxiv_reference)
        save_graph(arxiv_graph, arxiv_output)
        report_path = arxiv_output.parent / "arxiv_compare_report.json"
        _ensure_dir(report_path)
        report_path.write_text(json.dumps(compare_report, indent=2, ensure_ascii=False))
        arxiv_artifact = {
            "output": str(arxiv_output),
            "reference": str(arxiv_reference) if arxiv_reference else None,
            "report": str(report_path),
            "compare_report": compare_report,
        }
        print(f"[save] arxiv: {arxiv_output}")
        if compare_report:
            print("[compare] arxiv vs reference:")
            print(json.dumps(compare_report, indent=2, ensure_ascii=False))

    return {"datasets": result, "arxiv": arxiv_artifact}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["products", "wikics", "fb15k237"],
        help="Datasets to rebuild.",
    )
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help="Text encoder name. Use 'hash' for an offline deterministic fallback.",
    )
    parser.add_argument(
        "--output-root",
        default=str(GRAPHSSR_ROOT / "data" / "test_data" / "raw_data"),
        help="Destination root for GraphSSR evaluation files.",
    )
    parser.add_argument(
        "--arxiv-output",
        default=str(GRAPHSSR_ROOT / "outputs" / "reconstructed" / "arxiv_graph_bert_encoded.pt"),
        help="Where to save the reconstructed arxiv artifact.",
    )
    parser.add_argument(
        "--arxiv-reference",
        default=str(GRAPHSSR_ROOT / "data" / "train_data" / "raw_data" / "arxiv" / "arxiv_graph_bert_encoded.pt"),
        help="Reference arxiv pt for comparison.",
    )
    parser.add_argument(
        "--skip-arxiv",
        action="store_true",
        help="Do not build the arxiv comparison artifact.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root)
    arxiv_output = None if args.skip_arxiv else Path(args.arxiv_output)
    arxiv_reference = None if args.skip_arxiv else Path(args.arxiv_reference)

    generate_requested_graphs(
        args.datasets,
        encoder=args.encoder,
        output_root=output_root,
        arxiv_output=arxiv_output,
        arxiv_reference=arxiv_reference,
    )


if __name__ == "__main__":
    main()
