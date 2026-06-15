# Aanvraag toegang tot KMI-databronnen (ontwerp)

Niet-commercieel onderzoeksproject *Pluvio* — neerslagnowcasting.
Status: ontwerp, klaar om te versturen.

- **Aan:** opendata@meteo.be (of via het [contactformulier van het KMI](https://www.meteo.be/nl/over-het-kmi/contact))
- **Onderwerp:** Aanvraag toegang tot KMI-databronnen voor niet-commercieel onderzoek (neerslagnowcasting)

---

Geacht KMI Open Data-team,

Mijn naam is Jeroen Trappers. Ik werk aan *Pluvio*, een **niet-commercieel, openbronproject**
(gelicentieerd onder GPL-3.0) dat onderzoekt of de kortetermijnvoorspelling van neerslag — de
zogenaamde *nowcast* — verbeterd kan worden met een zelflerend model dat uitsluitend op open data
wordt getraind.

We maken op dit moment, met correcte bronvermelding (CC BY 4.0), reeds gebruik van uw publiek
toegankelijke diensten — onder meer het ALARO-model (`/service/alaro/wms`), het radar- en
synopproduct, en de 10-minuutwaarnemingen van de automatische weerstations — aangevuld met open
data van het KNMI en EUMETSAT.

Voor de verdere uitbouw van het onderzoek zouden enkele bijkomende KMI-datasets bijzonder
waardevol zijn. We hebben echter begrepen dat **registratie op het opendataportaal in principe
voorbehouden is aan onderzoeksinstellingen**. Pluvio is geen universiteit of erkende instelling,
maar wel een strikt niet-commercieel onderzoeksproject waarvan de code en resultaten openbaar
worden gedeeld. Onze eerste vraag is dan ook of toegang onder die voorwaarden mogelijk is, en zo
ja, onder welke modaliteiten.

Concreet zouden we graag toegang krijgen tot de volgende databronnen (in volgorde van belang
voor het project):

**1. Radargebaseerde neerslag (kern van het project)**
- **RADCLIM** — de historische radargebaseerde neerslagaccumulatie (Belgisch composiet, 1 km).
  Essentieel als trainings- en validatiebron aan Belgische zijde, complementair aan de
  Nederlandse KNMI-radar die we al gebruiken.
- **RADQPE** — de realtime radargebaseerde neerslagschatting (volgens het portaal nog niet
  publiek beschikbaar).

**2. Nowcasting & referentieproducten (als wetenschappelijke benchmark)**
- **INCA(-BE)** — uw eigen geblende analyse/nowcast van radar, NWP en waarnemingen. We zouden
  dit graag gebruiken als objectieve **referentie** om de prestaties van ons model
  wetenschappelijk mee te vergelijken.
- **Uurlijkse voorspellingen per gemeente** — eveneens nuttig als vergelijkingsbasis.

**3. Aanvullende waarnemingen**
- Volledige toegang tot de **AWS-waarnemingen** (10-min/uur/dag) via WMS/WFS.
- Het netwerk van **ceilometers/lidars** (wolkbasishoogte) — een nuttige indicator voor convectie.
- De **weerwaarschuwingen** voor België.
- Indien beschikbaar: **bliksemdetectiegegevens** — een sterke indicator voor actieve convectie,
  net het fenomeen dat met radarextrapolatie het moeilijkst te voorspellen is.

Indien het KMI nog andere datasets aanbiedt die relevant kunnen zijn voor neerslagnowcasting
(bv. polarimetrische of volumetrische radardata), vernemen we dat uiteraard ook graag.

Mogen we u vragen:
- of toegang tot (een deel van) deze datasets mogelijk is voor een niet-commercieel
  onderzoeksproject zoals het onze;
- onder welke voorwaarden, licentie en eventuele kosten dit kan;
- of een portaalaccount kan worden aangemaakt, dan wel of toegang via een andere weg verloopt.

Alle afgeleide producten en publicaties worden uiteraard voorzien van de gevraagde bronvermelding
("Koninklijk Meteorologisch Instituut van België – KMI/IRM"). Ik licht het project en het beoogde
gebruik graag verder toe, telefonisch of op een korte afspraak indien gewenst.

Alvast hartelijk dank voor uw tijd en antwoord.

Met vriendelijke groet,

Jeroen Trappers
*Pluvio — niet-commercieel onderzoeksproject neerslagnowcasting*
jeroen.trappers@secutec.com

---

## Notities (niet meesturen)

- **Eerlijkheid over de instelling.** Bewust niet voorgesteld alsof Pluvio een erkende
  onderzoeksinstelling is; de vraag staat open. Aanvragen via een universiteit/onderzoekspartner
  (of een onderzoeksafdeling binnen Secutec) vergroot de slaagkans aanzienlijk.
- **Het @secutec.com-adres** kan commercieel ogen — overweeg een persoonlijk/projectadres of
  expliciteer dat Secutec niet de begunstigde is.
- **Satelliet-WMS** weggelaten: al gedekt via EUMETSAT.
- Reeds geverifieerd (juni 2026): `/service/inca/wms` en `/service/satellite/wms` geven HTTP 403;
  RADCLIM e.a. zijn account-gated (download via `profile/api/v1/download-files`, Bearer-token).
