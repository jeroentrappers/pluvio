import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:latlong2/latlong.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/core/result/result.dart';
import 'package:pluvio/features/location/application/location_providers.dart';
import 'package:pluvio/features/location/data/location_service.dart';
import 'package:pluvio/features/radar/application/radar_providers.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';
import 'package:pluvio/features/radar/domain/radar_repository.dart';
import 'package:pluvio/features/radar/presentation/radar_screen.dart';
import 'package:pluvio/l10n/app_localizations.dart';

class _StubRepository implements RadarRepository {
  _StubRepository({required this.animation});

  final RadarAnimation animation;

  @override
  Future<Result<RadarAnimation, ApiFailure>> fetchAnimation(LatLng location) async {
    return Result.ok(animation);
  }
}

class _FakeLocationService implements LocationService {
  @override
  Future<LatLng?> currentLocation() async => const LatLng(50.85, 4.35);
}

class _NoopHttpClient implements HttpClient {
  @override
  dynamic noSuchMethod(Invocation invocation) => null;
}

void main() {
  final ref = DateTime.utc(2026, 5, 26, 8);
  final animation = RadarAnimation(
    frames: [
      RadarFrame(
        timestamp: ref,
        imageUrl: 'https://example.test/0.png',
        valueMmPerHour: 0,
      ),
      RadarFrame(
        timestamp: ref.add(const Duration(minutes: 10)),
        imageUrl: 'https://example.test/1.png',
        valueMmPerHour: 9,
      ),
    ],
    referenceTime: ref,
    location: const LatLng(50.85, 4.35),
  );

  // FlutterMap can't settle under TestWidgetsFlutterBinding (network tiles
  // never resolve), so this case is also exercised by integration_test/.
  testWidgets(
    'renders the headline derived from the animation',
    skip: true,
    (tester) async {
      await HttpOverrides.runZoned(
        () async {
          await tester.pumpWidget(
            ProviderScope(
              overrides: [
                locationServiceProvider.overrideWithValue(_FakeLocationService()),
                radarRepositoryProvider.overrideWithValue(
                  _StubRepository(animation: animation),
                ),
              ],
              child: MaterialApp(
                localizationsDelegates: AppLocalizations.localizationsDelegates,
                supportedLocales: AppLocalizations.supportedLocales,
                locale: const Locale('en'),
                home: const RadarScreen(),
              ),
            ),
          );
          for (var i = 0; i < 4; i++) {
            await tester.pump(const Duration(milliseconds: 50));
          }
          expect(find.text('Rain expected in 10 min.'), findsOneWidget);
          expect(find.byType(Slider), findsOneWidget);
        },
        createHttpClient: (_) => _NoopHttpClient(),
      );
    },
  );
}
