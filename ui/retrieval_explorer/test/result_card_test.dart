// result_card_test.dart — the structured card renders score, rank, badges and
// an expandable metadata table.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:retrieval_explorer/api_client.dart';
import 'package:retrieval_explorer/theme.dart';
import 'package:retrieval_explorer/widgets/result_card.dart';

Hit buildHit({String? document, String? body, int chars = 62}) => Hit.fromJson({
      'id': 'chunk-1',
      'rank': 1,
      'similarity': 0.9034,
      'distance': 0.0966,
      'document': document ??
          '[Informatika › Studije] Master studije informatike traju dve godine.',
      'body': body ?? 'Master studije informatike traju dve godine.',
      'metadata': {
        'page_url': 'https://imi.pmf.kg.ac.rs/informatika',
        'doc_kind': 'prose',
        'status': 'enrolling',
        'breadcrumb': 'Informatika › Studije',
        'keywords': 'master; espb',
      },
      'url': 'https://imi.pmf.kg.ac.rs/informatika',
      'breadcrumb': 'Informatika › Studije',
      'doc_kind': 'prose',
      'status': 'enrolling',
      'chars': chars,
      'words': 8,
    });

Widget wrap(Widget child) => MaterialApp(
      theme: buildTheme(),
      home: Scaffold(body: SingleChildScrollView(child: child)),
    );

void main() {
  testWidgets('shows rank, similarity, badges and the chunk text', (tester) async {
    await tester.pumpWidget(wrap(
      ResultCard(hit: buildHit(), terms: const ['master'], threshold: 0.7, k: 5),
    ));

    expect(find.text('#1'), findsOneWidget);
    expect(find.text('90.3%'), findsOneWidget);
    expect(find.text('Informatika › Studije'), findsOneWidget);
    expect(find.text('prose'), findsOneWidget);
    expect(find.text('enrolling'), findsOneWidget);
    expect(find.textContaining('rank 1 of 5'), findsOneWidget);
    expect(find.textContaining('distance 0.0966'), findsOneWidget);
    expect(find.textContaining('Master studije informatike traju dve godine.'), findsWidgets);
  });

  testWidgets('expands the full metadata table on demand', (tester) async {
    await tester.pumpWidget(wrap(
      ResultCard(hit: buildHit(), terms: const [], threshold: 0.7, k: 5),
    ));

    expect(find.text('page_url'), findsNothing);

    await tester.tap(find.textContaining('Show metadata'));
    await tester.pumpAndSettle();

    expect(find.text('page_url'), findsOneWidget);
    expect(find.text('doc_kind'), findsOneWidget);
    expect(find.textContaining('5 metadata fields'), findsOneWidget);

    await tester.tap(find.textContaining('Hide metadata'));
    await tester.pumpAndSettle();
    expect(find.text('page_url'), findsNothing);
  });

  testWidgets('expands the embedded text when it is longer than the preview', (tester) async {
    await tester.pumpWidget(wrap(
      ResultCard(
        hit: buildHit(document: 'x' * 900, body: 'y' * 900, chars: 900),
        terms: const [],
        threshold: 0.7,
        k: 5,
      ),
    ));

    expect(find.textContaining('Show full chunk'), findsOneWidget);
    await tester.tap(find.textContaining('Show full chunk'));
    await tester.pumpAndSettle();
    expect(find.textContaining('embedded text'), findsOneWidget);
  });
}
