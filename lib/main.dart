import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'app/app.dart';
import 'core/config/env.dart';
import 'core/logging/logger.dart';

Future<void> main() async {
  await runZonedGuarded<Future<void>>(
    () async {
      WidgetsFlutterBinding.ensureInitialized();
      await SystemChrome.setPreferredOrientations([
        DeviceOrientation.portraitUp,
        DeviceOrientation.portraitDown,
      ]);

      Env.assertValid();
      AppLogger.init();
      FlutterError.onError = AppLogger.captureFlutterError;
      PlatformDispatcher.instance.onError = (Object error, StackTrace stack) {
        AppLogger.captureException(error, stack);
        return true;
      };

      runApp(
        ProviderScope(
          observers: [AppLogger.riverpodObserver()],
          child: const PluvioApp(),
        ),
      );
    },
    (Object error, StackTrace stack) {
      AppLogger.captureException(error, stack);
    },
  );
}
