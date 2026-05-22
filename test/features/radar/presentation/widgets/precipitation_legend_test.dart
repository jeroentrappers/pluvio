import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/domain/nowcast.dart';
import 'package:pluvio/features/radar/presentation/widgets/precipitation_legend.dart';

void main() {
  testWidgets('legend shows one chip per non-none precipitation level',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: Scaffold(body: PrecipitationLegend()),
    ));
    // 4 levels: light, moderate, heavy, violent.
    expect(find.text('Light'), findsOneWidget);
    expect(find.text('Moderate'), findsOneWidget);
    expect(find.text('Heavy'), findsOneWidget);
    expect(find.text('Violent'), findsOneWidget);
  });

  test('palette returns distinct colours per level', () {
    final scheme = ColorScheme.fromSeed(seedColor: Colors.blue);
    final seen = <int>{};
    for (final level in PrecipitationLevel.values) {
      final c = PrecipitationPalette.of(level, scheme);
      seen.add(c.toARGB32());
    }
    expect(seen.length, PrecipitationLevel.values.length);
  });
}
