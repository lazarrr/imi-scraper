// live_api_test.dart — talks to a REAL running backend (tools/retrieval_api.py)
// and verifies the app-side parsing against the 747-chunk test collection.
//
//     python tools/retrieval_api.py --db-path data/local_test_store \
//            --collection university_knowledge_local_test --no-warm
//     cd ui/retrieval_explorer && flutter test test/live_api_test.dart
//
// The whole group is skipped when nothing is listening on the API port, so
// `flutter test` stays green offline (same idea as tests/test_deepseek_live.py).

import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:retrieval_explorer/api_client.dart';

const String apiBase = String.fromEnvironment(
  'IMI_API',
  defaultValue: 'http://127.0.0.1:8765',
);

Future<bool> backendIsUp() async {
  try {
    final uri = Uri.parse('$apiBase/api/health');
    final socket = await Socket.connect(uri.host, uri.port,
        timeout: const Duration(milliseconds: 800));
    socket.destroy();
    return true;
  } on SocketException {
    return false;
  }
}

void main() {
  late RetrievalApi api;

  setUpAll(() {
    // flutter_test blocks real sockets by default.
    HttpOverrides.global = null;
    api = RetrievalApi(baseUrl: apiBase);
  });

  test('the API answers with a store header', () async {
    if (!await backendIsUp()) {
      markTestSkipped('no retrieval API on $apiBase — start tools/retrieval_api.py');
      return;
    }
    final health = await api.health();
    expect(health.store.collection, isNotEmpty);
    expect(health.store.count, greaterThan(0));
    expect(health.store.embeddingModel, contains('e5'));
    expect(health.store.queryPrefix, 'query: ');
  }, timeout: const Timeout(Duration(minutes: 10)));

  test('a real query returns scored, structured results', () async {
    if (!await backendIsUp()) {
      markTestSkipped('no retrieval API on $apiBase');
      return;
    }
    final run = await api.search(
      query: 'Koliko traje master studije informatike?',
      k: 5,
      includeLegacy: false,
    );

    expect(run.results, isNotEmpty);
    expect(run.results.length, lessThanOrEqualTo(5));
    expect(run.counts.fetched, greaterThanOrEqualTo(run.results.length));

    var previous = 1.1;
    for (final hit in run.results) {
      expect(hit.rank, greaterThan(0));
      expect(hit.document, isNotEmpty);
      expect(hit.similarity, inInclusiveRange(run.threshold, 1.0));
      expect(hit.similarity, closeTo(1 - hit.distance, 0.0002));
      expect(hit.similarity, lessThanOrEqualTo(previous)); // descending
      previous = hit.similarity;
    }

    // Every rejection carries a machine-readable reason for the side panel.
    for (final rejected in run.rejected) {
      expect(rejected.reason, isNotNull);
      expect(
        const ['below_threshold', 'legacy_or_outdated', 'duplicate', 'beyond_k'],
        contains(rejected.reason),
      );
    }
    expect(run.counts.kept + run.counts.rejected, run.counts.fetched);
  }, timeout: const Timeout(Duration(minutes: 10)));

  test('include_legacy widens the candidate pool', () async {
    if (!await backendIsUp()) {
      markTestSkipped('no retrieval API on $apiBase');
      return;
    }
    final strict = await api.search(
        query: 'Predmeti na starom programu informatike 2011',
        k: 5,
        includeLegacy: false);
    // The query itself asks for a legacy programme, so the where-filter is
    // dropped automatically — this asserts the flag rounds-trips either way.
    final loose = await api.search(
        query: 'Predmeti na starom programu informatike 2011',
        k: 5,
        includeLegacy: true);
    expect(strict.whereFilter, isNull);
    expect(loose.includeLegacy, isTrue);
    expect(loose.results, isNotEmpty);
  }, timeout: const Timeout(Duration(minutes: 10)));
}
