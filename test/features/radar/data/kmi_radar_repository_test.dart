import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:latlong2/latlong.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/core/result/result.dart';
import 'package:pluvio/features/radar/data/kmi_radar_repository.dart';
import 'package:pluvio/features/radar/data/sources/kmi_nowcast_source.dart';
import 'package:pluvio/features/radar/data/sources/kmi_radar_source.dart';

import '../../../_helpers/load_fixture.dart';

class _MockDio extends Mock implements Dio {}

void main() {
  late _MockDio dio;
  late KmiRadarRepository repo;

  setUp(() {
    registerFallbackValue(Options());
    dio = _MockDio();
    repo = KmiRadarRepository(
      nowcastSource: KmiNowcastSource(dio: dio, baseUrl: 'https://app.example/'),
      radarSource: KmiRadarSource(
        dio: dio,
        wmsUrl: 'https://opendata.example/wms',
        layer: 'RADAR.BE_COMPOSITE',
      ),
      clock: () => DateTime.utc(2026, 5, 22, 10),
    );
  });

  group('fetchNowcast', () {
    test('maps DTO entries onto evenly-spaced NowcastPoints', () async {
      when(() => dio.get<Map<String, dynamic>>(
            any(),
            queryParameters: any(named: 'queryParameters'),
          )).thenAnswer(
        (_) async => Response<Map<String, dynamic>>(
          requestOptions: RequestOptions(path: ''),
          data: loadFixtureJson('kmi_nowcast_sample.json'),
          statusCode: 200,
        ),
      );

      final res = await repo.fetchNowcast(const LatLng(50.85, 4.35));
      expect(res.isOk, isTrue);
      final nowcast = res.valueOrNull!;
      expect(nowcast.points.length, 12);
      expect(
        nowcast.points[1].timestamp.difference(nowcast.points[0].timestamp),
        const Duration(minutes: 5),
      );
      expect(nowcast.minutesUntilRain, 10);
    });

    test('surfaces upstream failure unchanged', () async {
      when(() => dio.get<Map<String, dynamic>>(
            any(),
            queryParameters: any(named: 'queryParameters'),
          )).thenThrow(
        DioException(
          requestOptions: RequestOptions(path: ''),
          type: DioExceptionType.connectionError,
        ),
      );

      final res = await repo.fetchNowcast(const LatLng(0, 0));
      expect(res, isA<Err<dynamic, ApiFailure>>());
      expect(res.errorOrNull, isA<NetworkFailure>());
    });
  });

  group('fetchRadarAnimation', () {
    test('builds frames with TIME-bound tile URLs', () async {
      when(() => dio.get<String>(
            any(),
            queryParameters: any(named: 'queryParameters'),
            options: any(named: 'options'),
          )).thenAnswer(
        (_) async => Response<String>(
          requestOptions: RequestOptions(path: ''),
          data: loadFixtureString('kmi_radar_capabilities_sample.xml'),
          statusCode: 200,
        ),
      );

      final res = await repo.fetchRadarAnimation();
      expect(res.isOk, isTrue);
      final anim = res.valueOrNull!;
      expect(anim.frames.length, 9);
      expect(anim.frames.first.tileUrlTemplate, contains('time=2026-05-22'));
      // currentIndex should sit on or next to the clock-injected reference time.
      expect(
        anim.frames[anim.currentIndex].timestamp,
        DateTime.utc(2026, 5, 22, 10),
      );
    });
  });
}
