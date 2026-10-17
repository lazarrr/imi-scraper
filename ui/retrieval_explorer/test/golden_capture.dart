// golden_capture.dart — renders the real ExplorerPage (fed by captured API
// fixtures, no backend needed) into docs/retrieval_explorer*.png.
//
//     cd ui/retrieval_explorer
//     flutter test test/golden_capture.dart --update-goldens
//
// It is deliberately NOT named *_test.dart, so a plain `flutter test` skips it:
// re-running it rewrites checked-in images rather than asserting anything.
// System + Material icon fonts are loaded so the capture looks like the running
// app instead of the test framework's placeholder glyphs.

import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:retrieval_explorer/api_client.dart';
import 'package:retrieval_explorer/main.dart';
import 'package:retrieval_explorer/theme.dart';

const _window = Size(1500, 950);

Future<void> _loadFont(String family, String path) async {
  final file = File(path);
  if (!file.existsSync()) return;
  final bytes = file.readAsBytesSync();
  final loader = FontLoader(family)
    ..addFont(Future.value(ByteData.sublistView(bytes)));
  await loader.load();
}

Future<void> _loadFonts() async {
  await _loadFont('.AppleSystemUIFont', '/System/Library/Fonts/Supplemental/Arial.ttf');
  await _loadFont('Menlo', '/System/Library/Fonts/Supplemental/Courier New.ttf');
  final flutterRoot = Platform.environment['FLUTTER_ROOT'];
  if (flutterRoot != null) {
    await _loadFont(
      'MaterialIcons',
      '$flutterRoot/bin/cache/artifacts/material_fonts/MaterialIcons-Regular.otf',
    );
  }
}

http.Response _fixture(String name) => http.Response.bytes(
      File('test/fixtures/$name').readAsBytesSync(),
      200,
      headers: const {'content-type': 'application/json; charset=utf-8'},
    );

Future<void> _pumpExplorer(WidgetTester tester) async {
  final api = RetrievalApi(
    client: MockClient((request) async {
      switch (request.url.path) {
        case '/api/health':
          return _fixture('api_health.json');
        case '/api/search':
          return _fixture('api_search.json');
        case '/api/stats':
          return _fixture('api_stats.json');
        default:
          return http.Response('{"ok": false, "error": "not found"}', 404);
      }
    }),
  );

  await tester.pumpWidget(MaterialApp(
    title: 'Chroma Retrieval Explorer',
    debugShowCheckedModeBanner: false,
    theme: buildTheme(),
    home: ExplorerPage(api: api),
  ));

  // health → auto example query → results
  for (var i = 0; i < 6; i++) {
    await tester.pump(const Duration(milliseconds: 120));
  }
}

void main() {
  setUp(() async {
    await _loadFonts();
    // ignore: invalid_use_of_visible_for_testing_member
  });

  testWidgets('capture the result list', (tester) async {
    tester.view.physicalSize = _window * 2;
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await _pumpExplorer(tester);
    await expectLater(
      find.byType(ExplorerPage),
      matchesGoldenFile('../../../docs/retrieval_explorer.png'),
    );
  });

  testWidgets('capture an expanded card with its metadata table', (tester) async {
    tester.view.physicalSize = _window * 2;
    tester.view.devicePixelRatio = 2.0;
    addTearDown(tester.view.reset);

    await _pumpExplorer(tester);
    await tester.tap(find.textContaining('Show metadata').first);
    for (var i = 0; i < 4; i++) {
      await tester.pump(const Duration(milliseconds: 120));
    }
    await expectLater(
      find.byType(ExplorerPage),
      matchesGoldenFile('../../../docs/retrieval_explorer_metadata.png'),
    );
  });
}
