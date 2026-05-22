import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';
import 'package:pluvio/features/radar/presentation/widgets/timeline_slider.dart';
import 'package:pluvio/l10n/app_localizations.dart';

void main() {
  Widget host(Widget child) {
    return MaterialApp(
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      locale: const Locale('en'),
      home: Scaffold(body: child),
    );
  }

  testWidgets('shows the observation pill for frames at or before reference time',
      (tester) async {
    final ref = DateTime.utc(2026, 5, 22, 10);
    final anim = RadarAnimation(
      frames: [
        RadarFrame(
          timestamp: ref.subtract(const Duration(minutes: 5)),
          tileUrlTemplate: 'x',
        ),
        RadarFrame(timestamp: ref, tileUrlTemplate: 'x'),
      ],
      referenceTime: ref,
    );

    await tester.pumpWidget(host(
      TimelineSlider(animation: anim, currentIndex: 0, onChanged: (_) {}),
    ));

    expect(find.text('observation'), findsOneWidget);
    expect(find.text('forecast'), findsNothing);
  });

  testWidgets('shows the forecast pill for frames after reference time',
      (tester) async {
    final ref = DateTime.utc(2026, 5, 22, 10);
    final anim = RadarAnimation(
      frames: [
        RadarFrame(timestamp: ref, tileUrlTemplate: 'x'),
        RadarFrame(
          timestamp: ref.add(const Duration(minutes: 5)),
          tileUrlTemplate: 'x',
        ),
      ],
      referenceTime: ref,
    );

    await tester.pumpWidget(host(
      TimelineSlider(animation: anim, currentIndex: 1, onChanged: (_) {}),
    ));

    expect(find.text('forecast'), findsOneWidget);
  });

  testWidgets('emits the new index when the slider is dragged', (tester) async {
    final ref = DateTime.utc(2026, 5, 22, 10);
    final anim = RadarAnimation(
      frames: List.generate(
        5,
        (i) => RadarFrame(
          timestamp: ref.add(Duration(minutes: i * 5)),
          tileUrlTemplate: 'x',
        ),
      ),
      referenceTime: ref,
    );
    var captured = -1;

    await tester.pumpWidget(host(
      TimelineSlider(
        animation: anim,
        currentIndex: 0,
        onChanged: (v) => captured = v,
      ),
    ));

    await tester.drag(find.byType(Slider), const Offset(400, 0));
    await tester.pump();

    expect(captured, isNot(-1));
    expect(captured, greaterThan(0));
  });
}
