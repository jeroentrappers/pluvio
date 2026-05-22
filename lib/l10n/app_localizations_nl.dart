// ignore: unused_import
import 'package:intl/intl.dart' as intl;
import 'app_localizations.dart';

// ignore_for_file: type=lint

/// The translations for Dutch Flemish (`nl`).
class AppLocalizationsNl extends AppLocalizations {
  AppLocalizationsNl([String locale = 'nl']) : super(locale);

  @override
  String get appTitle => 'Pluvio';

  @override
  String get refresh => 'Vernieuwen';

  @override
  String get locationError => 'Kon je locatie niet bepalen.';

  @override
  String get radarError => 'Kon de radargegevens niet ophalen.';

  @override
  String get nowcastError => 'Kon de buienverwachting niet ophalen.';

  @override
  String get nowcastDry => 'Geen regen verwacht in de komende twee uur.';

  @override
  String get nowcastRaining => 'Het regent nu.';

  @override
  String nowcastRainInMinutes(int minutes) {
    final intl.NumberFormat minutesNumberFormat = intl.NumberFormat.decimalPattern(localeName);
    final String minutesString = minutesNumberFormat.format(minutes);

    return 'Regen verwacht over $minutesString min.';
  }
}
