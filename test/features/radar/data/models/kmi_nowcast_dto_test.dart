import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/data/models/kmi_nowcast_dto.dart';

import '../../../../_helpers/load_fixture.dart';

void main() {
  group('KmiNowcastDto.fromJson', () {
    test('parses the canonical KMI sample payload', () {
      final dto = KmiNowcastDto.fromJson(loadFixtureJson('kmi_nowcast_sample.json'));

      expect(dto.issuedAt, DateTime.utc(2026, 5, 22, 10));
      expect(dto.intervalMinutes, 5);
      expect(dto.precipitationMmPerHour.length, 12);
      expect(dto.precipitationMmPerHour.first, 0.0);
      expect(dto.precipitationMmPerHour[5], 3.0);
    });

    test('tolerates entries wrapped in objects with `value` or `intensity`', () {
      final dto = KmiNowcastDto.fromJson({
        'dateForecast': '2026-05-22T10:00:00Z',
        'nowcast': [
          0,
          {'value': 1.5},
          {'intensity': 2.0},
          'ignored-non-numeric',
        ],
      });

      expect(dto.precipitationMmPerHour, [0.0, 1.5, 2.0]);
      expect(dto.intervalMinutes, 5);
    });

    test('throws FormatException when dateForecast is missing', () {
      expect(
        () => KmiNowcastDto.fromJson({'nowcast': [0]}),
        throwsFormatException,
      );
    });

    test('throws FormatException when nowcast payload is missing', () {
      expect(
        () => KmiNowcastDto.fromJson({'dateForecast': '2026-05-22T10:00:00Z'}),
        throwsFormatException,
      );
    });
  });
}
