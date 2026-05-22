import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mocktail/mocktail.dart';
import 'package:pluvio/core/networking/api_failure.dart';
import 'package:pluvio/features/radar/data/sources/kmi_nowcast_source.dart';

import '../../../../_helpers/load_fixture.dart';

class _MockDio extends Mock implements Dio {}

void main() {
  late _MockDio dio;
  late KmiNowcastSource source;

  setUp(() {
    dio = _MockDio();
    source = KmiNowcastSource(dio: dio, baseUrl: 'https://example.test');
  });

  Response<Map<String, dynamic>> okResponse(Map<String, dynamic> body) {
    return Response<Map<String, dynamic>>(
      requestOptions: RequestOptions(path: ''),
      data: body,
      statusCode: 200,
    );
  }

  test('parses the success response into a DTO', () async {
    final fixture = loadFixtureJson('kmi_nowcast_sample.json');
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse(fixture));

    final result = await source.fetch(latitude: 50.85, longitude: 4.35);

    expect(result.isOk, isTrue);
    expect(result.valueOrNull!.precipitationMmPerHour.length, 12);
  });

  test('returns NetworkFailure on Dio connection error', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(
      DioException(
        requestOptions: RequestOptions(path: ''),
        type: DioExceptionType.connectionError,
      ),
    );

    final result = await source.fetch(latitude: 0, longitude: 0);
    expect(result.errorOrNull, isA<NetworkFailure>());
  });

  test('returns ServerFailure with status on 5xx', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenThrow(
      DioException(
        requestOptions: RequestOptions(path: ''),
        type: DioExceptionType.badResponse,
        response: Response<dynamic>(
          requestOptions: RequestOptions(path: ''),
          statusCode: 503,
        ),
      ),
    );

    final result = await source.fetch(latitude: 0, longitude: 0);
    final failure = result.errorOrNull;
    expect(failure, isA<ServerFailure>());
    expect((failure! as ServerFailure).statusCode, 503);
  });

  test('returns ParseFailure when payload shape is unexpected', () async {
    when(() => dio.get<Map<String, dynamic>>(
          any(),
          queryParameters: any(named: 'queryParameters'),
        )).thenAnswer((_) async => okResponse({'unrelated': 'shape'}));

    final result = await source.fetch(latitude: 0, longitude: 0);
    expect(result.errorOrNull, isA<ParseFailure>());
  });
}
