import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/features/radar/domain/nowcast.dart';

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

  group('Nowcast.minutesUntilRain', () {
    final issued = DateTime.utc(2026, 5, 22, 10);
    const brussels = LatLng(50.85, 4.35);

    NowcastPoint pt(int minutes, double mm) => NowcastPoint(
          timestamp: issued.add(Duration(minutes: minutes)),
          precipitationMmPerHour: mm,
        );

    test('returns null when the whole horizon is dry', () {
      final n = Nowcast(
        location: brussels,
        issuedAt: issued,
        points: [pt(0, 0), pt(5, 0), pt(10, 0)],
      );
      expect(n.hasRain, isFalse);
      expect(n.minutesUntilRain, isNull);
    });

    test('returns minutes until the first rain timestep', () {
      final n = Nowcast(
        location: brussels,
        issuedAt: issued,
        points: [pt(0, 0), pt(5, 0), pt(10, 1.2), pt(15, 3.4)],
      );
      expect(n.hasRain, isTrue);
      expect(n.minutesUntilRain, 10);
    });

    test('returns 0 when it is already raining at issue time', () {
      final n = Nowcast(
        location: brussels,
        issuedAt: issued,
        points: [pt(0, 0.5), pt(5, 0.6)],
      );
      expect(n.minutesUntilRain, 0);
    });
  });
}
