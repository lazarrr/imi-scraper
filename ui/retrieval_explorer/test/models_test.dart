// models_test.dart — parsing of the retrieval API payload + query highlighting.
//
// The JSON below is a trimmed copy of a real `/api/search` response, so the
// test fails if `src/core/retrieval_service.py` ever renames a field.

import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:retrieval_explorer/api_client.dart';
import 'package:retrieval_explorer/widgets/common.dart';

const String sampleJson = '''
{
  "ok": true,
  "query": "Koliko traje master informatika?",
  "normalized_query": "koliko traje master informatika?",
  "k": 5,
  "threshold": 0.7,
  "include_legacy": false,
  "where_filter": {"\$and": [{"status": {"\$ne": "legacy"}}]},
  "store": {
    "path": "/tmp/data/vector_store",
    "collection": "university_knowledge",
    "count": 747,
    "embedding_model": "intfloat/multilingual-e5-large",
    "distance": "cosine",
    "query_prefix": "query: ",
    "passage_prefix": "passage: ",
    "threshold": 0.7,
    "fetch_k": 20,
    "collections": [
      {"name": "university_knowledge", "count": 747, "active": true},
      {"name": "demo_smoke_test", "count": 8, "active": false}
    ]
  },
  "timing": {"embed_ms": 121.5, "query_ms": 8.2, "total_ms": 132.1},
  "counts": {
    "fetched": 20, "kept": 5, "rejected": 15, "below_threshold": 9,
    "legacy_or_outdated": 4, "duplicate": 1, "beyond_k": 1
  },
  "results": [
    {
      "id": "chunk-1", "rank": 1, "similarity": 0.9034, "distance": 0.0966,
      "document": "[Informatika › Studije] Master studije informatike traju dve godine.",
      "body": "Master studije informatike traju dve godine.",
      "metadata": {"page_url": "https://imi.pmf.kg.ac.rs/informatika", "doc_kind": "prose",
                   "status": "enrolling", "breadcrumb": "Informatika › Studije", "keywords": "master; espb"},
      "url": "https://imi.pmf.kg.ac.rs/informatika",
      "breadcrumb": "Informatika › Studije", "page_title": "Informatika",
      "doc_kind": "prose", "status": "enrolling", "chars": 62, "words": 8,
      "chips": [{"key": "keywords", "value": "master; espb"}]
    }
  ],
  "rejected": [
    {
      "id": "chunk-9", "similarity": 0.6120, "distance": 0.3880,
      "document": "[Stari program] Predmeti iz 2011.", "body": "Predmeti iz 2011.",
      "metadata": {"status": "legacy"}, "doc_kind": "table", "status": "legacy",
      "url": "https://imi.pmf.kg.ac.rs/stari-program", "breadcrumb": "Stari program",
      "reason": "legacy_or_outdated",
      "reason_label": "Legacy / outdated version",
      "reason_detail": "status=legacy (ask about an older program to include it)"
    }
  ]
}
''';

void main() {
  group('SearchRun.fromJson', () {
    final run = SearchRun.fromJson(jsonDecode(sampleJson) as Map<String, dynamic>);

    test('reads the store header', () {
      expect(run.store.collection, 'university_knowledge');
      expect(run.store.count, 747);
      expect(run.store.collections.length, 2);
      expect(run.store.collections.first.active, isTrue);
    });

    test('reads timings and the candidate funnel', () {
      expect(run.totalMs, 132.1);
      expect(run.counts.fetched, 20);
      expect(run.counts.legacyOrOutdated, 4);
      expect(run.counts.kept, 5);
    });

    test('keeps distance and similarity as separate numbers', () {
      final hit = run.results.single;
      expect(hit.rank, 1);
      expect(hit.similarity, closeTo(0.9034, 0.0001));
      expect(hit.distance, closeTo(0.0966, 0.0001));
      expect(hit.title, 'Informatika › Studije');
      expect(hit.metadataRows.length, 5);
    });

    test('exposes rejection reasons for the filtered-out panel', () {
      final rejected = run.rejected.single;
      expect(rejected.reason, 'legacy_or_outdated');
      expect(rejected.reasonLabel, isNotEmpty);
      expect(rejected.reasonDetail, contains('status=legacy'));
      expect(rejected.isLegacy, isTrue);
    });
  });

  group('query highlighting', () {
    test('tokenises words of three or more characters', () {
      expect(queryTerms('Koliko traje master informatika?'),
          ['koliko', 'traje', 'master', 'informatika']);
      expect(queryTerms('a je u'), isEmpty);
    });

    test('folds Serbian diacritics without changing length', () {
      expect(foldAscii('Školarina za studente'), 'Skolarina za studente');
      expect(foldAscii('Školarina').length, 'Školarina'.length);
    });

    test('marks every occurrence of a term, diacritics included', () {
      final spans = highlightSpans(
        'Školarina za samofinansirajuće studente',
        ['skolarina'],
        base: const TextStyle(),
        hit: const TextStyle(fontWeight: FontWeight.bold),
      );
      expect(spans.first.text, 'Školarina');
      expect(spans.first.style?.fontWeight, FontWeight.bold);
      expect(spans.length, 2);
    });
  });
}
