import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/features/radar/data/kmi_radar_repository.dart';
import 'package:pluvio/features/radar/data/sources/kmi_app_api_source.dart';
import 'package:pluvio/features/radar/domain/radar_animation.dart';

import '../../../_helpers/load_fixture.dart';

class _MockDio extends Mock implements Dio {}

void main() {
  late _MockDio dio;
  late KmiRadarRepository repo;

  // Pick a clock that sits inside the fixture's frame range so currentIndex
  // lands on a known frame.
  final clock = DateTime.utc(2026, 5, 26, 8); // first frame in the fixture

  setUp(() {
    dio = _MockDio();
    repo = KmiRadarRepository(
      source: KmiAppApiSource(
        dio: dio,
        baseUrl: 'https://app.example/',
        signingClock: () => clock,
      ),
      clock: () => clock,
    );
  });

  test('produces a RadarAnimation whose frames preserve the wire ordering', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => Response<Map<String, dynamic>>(
          requestOptions: RequestOptions(path: ''),
          data: loadFixtureJson('kmi_get_forecasts_sample.json'),
          statusCode: 200,
        ));

    final res = await repo.fetchAnimation(const LatLng(50.85, 4.35));

    expect(res.isOk, isTrue);
    final anim = res.valueOrNull!;
    expect(anim.frames.length, 30);
    expect(anim.location, const LatLng(50.85, 4.35));

    // Frames are sorted ascending by timestamp.
    for (var i = 1; i < anim.frames.length; i++) {
      expect(
        anim.frames[i].timestamp.isAfter(anim.frames[i - 1].timestamp),
        isTrue,
      );
    }
  });

  test('converts mm/10min from the wire to mm/h on the domain frames', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => Response<Map<String, dynamic>>(
          requestOptions: RequestOptions(path: ''),
          data: <String, dynamic>{
            'cityName': 'Test',
            'country': 'BE',
            'animation': {
              'type': '10min',
              'sequence': [
                {
                  'time': '2026-05-26T08:00:00+02:00',
                  'uri': 'https://cdn.example/0.png',
                  'value': 1.5, // mm/10min → 9 mm/h
                },
              ],
            },
          },
          statusCode: 200,
        ));

    final res = await repo.fetchAnimation(const LatLng(50.85, 4.35));
    final frame = res.valueOrNull!.frames.single;
    expect(frame.valueMmPerHour, 9.0);
    expect(frame.level, PrecipitationLevel.heavy);
  });

  test('surfaces upstream failures unchanged', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(DioException(
      requestOptions: RequestOptions(path: ''),
      type: DioExceptionType.connectionError,
    ));

    final res = await repo.fetchAnimation(const LatLng(0, 0));
    expect(res.errorOrNull, isA<NetworkFailure>());
  });
}
