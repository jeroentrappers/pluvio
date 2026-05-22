// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for German (`de`).
class AppLocalizationsDe extends AppLocalizations {
  AppLocalizationsDe([String locale = 'de']) : super(locale);

  @override
  String get appTitle => 'Pluvio';

  @override
  String get refresh => 'Aktualisieren';

  @override
  String get locationError => 'Standort konnte nicht ermittelt werden.';

  @override
  String get radarError => 'Radardaten konnten nicht geladen werden.';

  @override
  String get nowcastError => 'Nowcast konnte nicht geladen werden.';

  @override
  String get nowcastDry => 'In den nächsten zwei Stunden kein Regen erwartet.';

  @override
  String get nowcastRaining => 'Es regnet jetzt.';

  @override
  String nowcastRainInMinutes(int minutes) {
    final intl.NumberFormat minutesNumberFormat = intl.NumberFormat.decimalPattern(localeName);
    final String minutesString = minutesNumberFormat.format(minutes);

    return 'Regen in $minutesString Min. erwartet.';
  }
}
