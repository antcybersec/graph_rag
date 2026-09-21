# Architecture

## System

```mermaid
flowchart TB
  subgraph ING["Ingestion — one-time"]
    CORPUS["corpus.jsonl<br/>2,951 Wikipedia articles"]
    CORPUS --> CHUNK["chunk + embed<br/>BGE, local, no API"]
    CORPUS --> EXTRACT["entity + relation extraction<br/>LLM, one call per document"]
    CORPUS --> INFOBOX["infobox parser<br/>deterministic, no LLM"]
  end

  subgraph TG["TigerGraph Savanna"]
    DOC["Document — Chunk<br/>vector index"]
    ENT["Entity — RELATED_TO<br/>12k entities"]
    EVENT["OlympicEvent — Games — Sport — Venue<br/>PREV_GAMES · 2,210 events"]
    FACT["TemporalFact<br/>2,316 interval-stamped facts"]
  end

  CHUNK --> DOC
  EXTRACT --> ENT
  INFOBOX --> EVENT
  INFOBOX --> FACT

  subgraph Q["Query time"]
    RAG["1 · RAG<br/>vector search → answer"]
    GRAG["2 · GraphRAG<br/>entity link → 1-hop → answer"]
    AGENT["3 · Agentic GraphRAG — the Investigator<br/>orchestrator picks the next action"]
    JEV["4 · Jev planner<br/>typed selection, no generation model"]
  end

  subgraph TOOLS["Tools the orchestrator chooses between"]
    T1["query_events<br/>structured GSQL: count, argmax, lookup"]
    T2["query_facts<br/>as-of date, timeline"]
    T3["link_entities"]
    T4["graph_traverse"]
    T5["search_chunks<br/>vector"]
  end

  RAG --> DOC
  GRAG --> ENT
  AGENT --> T1 & T2 & T3 & T4 & T5
  T1 --> EVENT
  T2 --> FACT
  T3 --> ENT
  T4 --> ENT
  T5 --> DOC
  JEV --> T1

  AGENT --> EVID["Evidence + provenance<br/>doc id · source URL · originating query"]
  JEV --> EVID
  EVID --> ANS["Answer + citations"]
  EVID --> VERIFY["Verification pass<br/>supports / contradicts / says_nothing"]
```

## How the agent decides a step

```mermaid
flowchart LR
  Q["question"] --> P["orchestrator call<br/>plan + evidence_check<br/>in ONE structured call"]
  P -->|"tool"| R["run tool"]
  R --> E["add evidence<br/>with provenance"]
  E --> P
  P -->|"evidence sufficient"| A["answer with citations"]
  P -->|"budget exhausted"| F["best effort, gaps named"]
```

The self-evaluation rides in the same call as the routing decision, so
evaluating the evidence costs no extra LLM call. Measured over 100 questions:
2.04 steps, and all 100 investigations stopped because the agent judged the
evidence sufficient, never because it hit the iteration cap.

## The two answer paths, and why both exist

```mermaid
flowchart LR
  subgraph GEN["Agentic path — needs a generation model"]
    G1["orchestrator plans"] --> G2["tools"] --> G3["answer written by the model"]
  end
  subgraph SEL["Selection path — no generation model"]
    S1["Choice: operation · sport · Games"] --> S2["regex: numbers, dates, venue"]
    S2 --> S3["Choice: event or venue, narrowed"]
    S3 --> S4["GSQL computes; answer is a template"]
  end
```

Both reach 99/100. The agentic path generalises to anything in the corpus; the
selection path is 10x faster, makes no generation calls, and therefore cannot
fail on a rate limit — which is the difference between a demo that runs and one
that dies on stage.
