import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/features/radar/data/sources/kmi_app_api_source.dart';

import '../../../../_helpers/load_fixture.dart';

class _MockDio extends Mock implements Dio {}

void main() {
  late _MockDio dio;
  late KmiAppApiSource source;

  setUp(() {
    dio = _MockDio();
    source = KmiAppApiSource(
      dio: dio,
      baseUrl: 'https://app.example/',
      signingClock: () => DateTime(2026, 5, 26, 12),
    );
  });

  Response<Map<String, dynamic>> okResponse(Map<String, dynamic> body) {
    return Response<Map<String, dynamic>>(
      requestOptions: RequestOptions(path: ''),
      data: body,
      statusCode: 200,
    );
  }

  test('signs every call with a daily-rotating key', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse(
              loadFixtureJson('kmi_get_forecasts_sample.json'),
            ));

    await source.fetch(latitude: 50.85, longitude: 4.35);

    final captured = verify(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: captureAny(named: 'queryParameters'),
        )).captured.single as Map<String, dynamic>;
    expect(captured['s'], 'getForecasts');
    expect(captured['k'], matches(RegExp(r'^[a-f0-9]{32}$')));
    expect(captured['lat'], 50.85);
    expect(captured['long'], 4.35);
  });

  test('parses the real fixture into a DTO', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse(
              loadFixtureJson('kmi_get_forecasts_sample.json'),
            ));

    final result = await source.fetch(latitude: 50.85, longitude: 4.35);

    expect(result.isOk, isTrue);
    final dto = result.valueOrNull!;
    expect(dto.cityName, 'Brussels');
    expect(dto.animation.sequence, isNotEmpty);
  });

  test('returns NetworkFailure on Dio connection error', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(DioException(
      requestOptions: RequestOptions(path: ''),
      type: DioExceptionType.connectionError,
    ));

    final result = await source.fetch(latitude: 0, longitude: 0);
    expect(result.errorOrNull, isA<NetworkFailure>());
  });

  test('returns ServerFailure with status on 5xx', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(DioException(
      requestOptions: RequestOptions(path: ''),
      type: DioExceptionType.badResponse,
      response: Response<dynamic>(
        requestOptions: RequestOptions(path: ''),
        statusCode: 503,
      ),
    ));

    final result = await source.fetch(latitude: 0, longitude: 0);
    expect(result.errorOrNull, isA<ServerFailure>());
    expect((result.errorOrNull! as ServerFailure).statusCode, 503);
  });

  test('returns ParseFailure when the response is the wrong shape', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse({'unrelated': 'shape'}));

    final result = await source.fetch(latitude: 0, longitude: 0);
    expect(result.errorOrNull, isA<ParseFailure>());
  });

  test('rounds coordinates to 6 decimals', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse(
              loadFixtureJson('kmi_get_forecasts_sample.json'),
            ));

    await source.fetch(latitude: 50.8503123456789, longitude: 4.3517123456789);

    final captured = verify(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: captureAny(named: 'queryParameters'),
        )).captured.single as Map<String, dynamic>;
    expect(captured['lat'], 50.850312);
    expect(captured['long'], 4.351712);
  });
}
