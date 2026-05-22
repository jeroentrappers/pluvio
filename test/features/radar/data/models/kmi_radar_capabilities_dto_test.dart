import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/data/models/kmi_radar_capabilities_dto.dart';

void main() {
  group('KmiRadarCapabilitiesDto.fromTimeDimension', () {
    test('parses a comma-separated list of ISO timestamps', () {
      final dto = KmiRadarCapabilitiesDto.fromTimeDimension(
        '2026-05-22T09:55:00Z, 2026-05-22T10:00:00Z, 2026-05-22T10:05:00Z',
      );
      expect(dto.timeSteps.length, 3);
      expect(dto.timeSteps.first, DateTime.utc(2026, 5, 22, 9, 55));
      expect(dto.timeSteps.last, DateTime.utc(2026, 5, 22, 10, 5));
    });

    test('expands a start/end/period range', () {
      final dto = KmiRadarCapabilitiesDto.fromTimeDimension(
        '2026-05-22T10:00:00Z/2026-05-22T10:20:00Z/PT5M',
      );
      expect(dto.timeSteps.length, 5);
      expect(dto.timeSteps[2], DateTime.utc(2026, 5, 22, 10, 10));
    });

    test('skips unparseable tokens but keeps the valid ones', () {
      final dto = KmiRadarCapabilitiesDto.fromTimeDimension(
        '2026-05-22T10:00:00Z, garbage, 2026-05-22T10:05:00Z',
      );
      expect(dto.timeSteps.length, 2);
    });

    test('returns an empty list for empty input', () {
      final dto = KmiRadarCapabilitiesDto.fromTimeDimension('');
      expect(dto.timeSteps, isEmpty);
    });
  });
}
