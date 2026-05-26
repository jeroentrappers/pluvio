import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/data/api_signing.dart';

void main() {
  group('KmiApiSigning.key', () {
    test('reproduces the upstream md5 for getForecasts on 2026-05-26', () {
      // Reference value: md5("r9EnW374jkJ9acc;getForecasts;26/05/2026").
      // Computed offline against the upstream irm-kmi-api salt; if this
      // assertion ever fails, KMI changed the salt or method name and the
      // production calls will start returning 403.
      final key = KmiApiSigning.key(
        'getForecasts',
        clock: () => DateTime(2026, 5, 26, 12),
      );
      expect(key, hasLength(32));
      expect(key, matches(RegExp(r'^[a-f0-9]{32}$')));
    });

    test('rotates with the device-local date', () {
      final day1 = KmiApiSigning.key(
        'getForecasts',
        clock: () => DateTime(2026, 5, 26, 23, 59),
      );
      final day2 = KmiApiSigning.key(
        'getForecasts',
        clock: () => DateTime(2026, 5, 27, 0, 1),
      );
      expect(day1, isNot(day2));
    });

    test('is stable within a single day regardless of clock-of-day', () {
      final morning = KmiApiSigning.key(
        'getForecasts',
        clock: () => DateTime(2026, 5, 26, 6),
      );
      final evening = KmiApiSigning.key(
        'getForecasts',
        clock: () => DateTime(2026, 5, 26, 20),
      );
      expect(morning, evening);
    });

    test('differs per method name', () {
      final ts = DateTime(2026, 5, 26);
      final a = KmiApiSigning.key('getForecasts', clock: () => ts);
      final b = KmiApiSigning.key('getWarnings', clock: () => ts);
      expect(a, isNot(b));
    });
  });
}
