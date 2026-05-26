// Replaced by feature-scoped tests under test/features/. Kept so the default
// `flutter test` entry point still asserts something useful.

import 'package:flutter_test/flutter_test.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';

void main() {
  test('PrecipitationLevel.fromMmPerHour matches the WMO classification', () {
    expect(PrecipitationLevel.fromMmPerHour(0), PrecipitationLevel.none);
    expect(PrecipitationLevel.fromMmPerHour(1), PrecipitationLevel.light);
    expect(PrecipitationLevel.fromMmPerHour(3), PrecipitationLevel.moderate);
    expect(PrecipitationLevel.fromMmPerHour(10), PrecipitationLevel.heavy);
    expect(PrecipitationLevel.fromMmPerHour(60), PrecipitationLevel.violent);
  });
}
