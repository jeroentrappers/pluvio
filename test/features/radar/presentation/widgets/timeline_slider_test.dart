import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';
import 'package:pluvio/features/radar/presentation/widgets/timeline_slider.dart';
import 'package:pluvio/l10n/app_localizations.dart';

void main() {
  const here = LatLng(50.85, 4.35);

  Widget host(Widget child) {
    return MaterialApp(
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      locale: const Locale('en'),
      home: Scaffold(body: child),
    );
  }

  RadarFrame f(DateTime t, [double mm = 0]) => RadarFrame(
        timestamp: t,
        imageUrl: 'https://example.test/${t.millisecondsSinceEpoch}.png',
        valueMmPerHour: mm,
      );

  testWidgets('shows the observation pill for frames at or before reference time',
      (tester) async {
    final ref = DateTime.utc(2026, 5, 26, 8);
    final anim = RadarAnimation(
      frames: [
        f(ref.subtract(const Duration(minutes: 10))),
        f(ref),
      ],
      referenceTime: ref,
      location: here,
    );

    await tester.pumpWidget(host(
      TimelineSlider(
        animation: anim,
        currentIndex: 0,
        onChanged: (_) {},
        isPlaying: false,
        onPlayPause: () {},
      ),
    ));

    expect(find.text('observation'), findsOneWidget);
    expect(find.text('forecast'), findsNothing);
  });

  testWidgets('shows the forecast pill for frames after reference time',
      (tester) async {
    final ref = DateTime.utc(2026, 5, 26, 8);
    final anim = RadarAnimation(
      frames: [
        f(ref),
        f(ref.add(const Duration(minutes: 10))),
      ],
      referenceTime: ref,
      location: here,
    );

    await tester.pumpWidget(host(
      TimelineSlider(
        animation: anim,
        currentIndex: 1,
        onChanged: (_) {},
        isPlaying: false,
        onPlayPause: () {},
      ),
    ));

    expect(find.text('forecast'), findsOneWidget);
  });

  testWidgets('emits the new index when the slider is dragged', (tester) async {
    final ref = DateTime.utc(2026, 5, 26, 8);
    final anim = RadarAnimation(
      frames: List.generate(
        5,
        (i) => f(ref.add(Duration(minutes: i * 10))),
      ),
      referenceTime: ref,
      location: here,
    );
    var captured = -1;

    await tester.pumpWidget(host(
      TimelineSlider(
        animation: anim,
        currentIndex: 0,
        onChanged: (v) => captured = v,
        isPlaying: false,
        onPlayPause: () {},
      ),
    ));

    await tester.drag(find.byType(Slider), const Offset(400, 0));
    await tester.pump();

    expect(captured, isNot(-1));
    expect(captured, greaterThan(0));
  });
}
