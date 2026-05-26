import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';

void main() {
  group('PrecipitationLevel.fromMmPerHour', () {
    test('returns none for zero or negative values', () {
      expect(PrecipitationLevel.fromMmPerHour(0), PrecipitationLevel.none);
      expect(PrecipitationLevel.fromMmPerHour(-1), PrecipitationLevel.none);
    });

    test('matches the WMO 1985 buckets at the boundaries', () {
      expect(PrecipitationLevel.fromMmPerHour(2.49), PrecipitationLevel.light);
      expect(PrecipitationLevel.fromMmPerHour(2.5), PrecipitationLevel.moderate);
      expect(PrecipitationLevel.fromMmPerHour(7.49), PrecipitationLevel.moderate);
      expect(PrecipitationLevel.fromMmPerHour(7.5), PrecipitationLevel.heavy);
      expect(PrecipitationLevel.fromMmPerHour(49.9), PrecipitationLevel.heavy);
      expect(PrecipitationLevel.fromMmPerHour(50), PrecipitationLevel.violent);
    });
  });

  group('RadarAnimation', () {
    final ref = DateTime.utc(2026, 5, 26, 8);
    const brussels = LatLng(50.85, 4.35);

    RadarFrame f(int minutesFromRef, double mmPerHour) => RadarFrame(
          timestamp: ref.add(Duration(minutes: minutesFromRef)),
          imageUrl: 'https://cdn.example/$minutesFromRef.png',
          valueMmPerHour: mmPerHour,
        );

    test('minutesUntilRain ignores past frames', () {
      final anim = RadarAnimation(
        frames: [
          f(-20, 5),    // past rain — ignore
          f(-10, 3),    // past rain — ignore
          f(0, 0),      // now, dry
          f(10, 0),
          f(20, 1.5),   // first future rain
        ],
        referenceTime: ref,
        location: brussels,
      );
      expect(anim.minutesUntilRain, 20);
      expect(anim.hasRainAhead, isTrue);
    });

    test('minutesUntilRain returns 0 when raining at reference time', () {
      final anim = RadarAnimation(
        frames: [f(-10, 0), f(0, 1.2), f(10, 0)],
        referenceTime: ref,
        location: brussels,
      );
      expect(anim.minutesUntilRain, 0);
    });

    test('minutesUntilRain returns null when the future horizon is dry', () {
      final anim = RadarAnimation(
        frames: [f(-10, 4), f(0, 0), f(10, 0), f(20, 0)],
        referenceTime: ref,
        location: brussels,
      );
      expect(anim.hasRainAhead, isFalse);
      expect(anim.minutesUntilRain, isNull);
    });

    test('currentIndex picks the frame closest to reference time', () {
      final anim = RadarAnimation(
        frames: [f(-20, 0), f(-10, 0), f(2, 0), f(15, 0)],
        referenceTime: ref,
        location: brussels,
      );
      // 2 minutes after ref is the closest.
      expect(anim.frames[anim.currentIndex].timestamp,
          ref.add(const Duration(minutes: 2)));
    });
  });
}
