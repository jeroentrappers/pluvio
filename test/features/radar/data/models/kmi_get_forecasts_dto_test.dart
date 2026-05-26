import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/data/models/kmi_get_forecasts_dto.dart';

import '../../../../_helpers/load_fixture.dart';

void main() {
  group('KmiGetForecastsDto.fromJson', () {
    test('parses the redacted real getForecasts response', () {
      final dto = KmiGetForecastsDto.fromJson(
        loadFixtureJson('kmi_get_forecasts_sample.json'),
      );

      expect(dto.cityName, 'Brussels');
      expect(dto.country, 'BE');
      expect(dto.animation.sequence, isNotEmpty);
      expect(dto.animation.interval, const Duration(minutes: 10));
    });

    test('parses the first animation frame correctly', () {
      final dto = KmiGetForecastsDto.fromJson(
        loadFixtureJson('kmi_get_forecasts_sample.json'),
      );
      final first = dto.animation.sequence.first;

      expect(first.time.isUtc, isTrue);
      expect(first.imageUrl, startsWith('https://cdn.meteo.be'));
      expect(first.valueMmPer10Min, isA<double>());
    });

    test('throws FormatException when animation block is missing', () {
      expect(
        () => KmiGetForecastsDto.fromJson({'cityName': 'Brussels'}),
        throwsFormatException,
      );
    });

    test('throws FormatException when sequence is missing', () {
      expect(
        () => KmiGetForecastsDto.fromJson({
          'cityName': 'Brussels',
          'animation': {'type': '10min'},
        }),
        throwsFormatException,
      );
    });

    test('skips non-map entries in sequence rather than throwing', () {
      final dto = KmiGetForecastsDto.fromJson({
        'animation': {
          'type': '10min',
          'sequence': [
            {
              'time': '2026-05-26T08:00:00+02:00',
              'uri': 'https://cdn.example/0.png',
              'value': 0.5,
            },
            'junk',
            42,
            {
              'time': '2026-05-26T08:10:00+02:00',
              'uri': 'https://cdn.example/1.png',
              'value': 1,
            },
          ],
        },
      });
      expect(dto.animation.sequence, hasLength(2));
    });

    test('defaults missing or non-numeric values to 0', () {
      final dto = KmiGetForecastsDto.fromJson({
        'animation': {
          'sequence': [
            {
              'time': '2026-05-26T08:00:00+02:00',
              'uri': 'https://cdn.example/0.png',
              // no value
            },
            {
              'time': '2026-05-26T08:10:00+02:00',
              'uri': 'https://cdn.example/1.png',
              'value': 'NaN-as-string',
            },
          ],
        },
      });
      expect(dto.animation.sequence.first.valueMmPer10Min, 0);
      expect(dto.animation.sequence.last.valueMmPer10Min, 0);
    });
  });
}
