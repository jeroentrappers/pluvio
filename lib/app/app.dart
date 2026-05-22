import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../features/radar/presentation/radar_screen.dart';
import '../l10n/app_localizations.dart';
import 'theme.dart';

class PluvioApp extends ConsumerWidget {
  const PluvioApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp(
      title: 'Pluvio',
      debugShowCheckedModeBanner: false,
      theme: PluvioTheme.light(),
      darkTheme: PluvioTheme.dark(),
      themeMode: ThemeMode.system,
      localizationsDelegates: AppLocalizations.localizationsDelegates,
      supportedLocales: AppLocalizations.supportedLocales,
      home: const RadarScreen(),
    );
  }
}
