from relbench.base import TaskType

from .ctu_entity_task import CTUEntityTask
from .ctu_link_task import CTULinkTask

# fmt: off
__all__ = [
    "AccidentsOriginalTask", "AdventureWorksOriginalTask", "AirlineOriginalTask",
    "AtherosclerosisOriginalTask", "BasketballMenOriginalTask",
    "BasketballWomenOriginalTask", "BiodegradabilityOriginalTask",
    "BupaOriginalTask", "CarcinogenesisOriginalTask", "CDESchoolsOriginalTask", 
    "ChessOriginalTask", "ClassicModelsOriginalTask", "CORAOriginalTask", 
    "CountriesOriginalTask", "CraftBeerOriginalTask", "CreditOriginalTask", 
    "DallasOriginalTask", "DCGOriginalTask", "DiabetesOriginalTask",
    "DunurOriginalTask", "EltiOriginalTask", "EmployeeOriginalTask", 
    "ErgastF1OriginalTask", "ExpendituresOriginalTask", "EmployeeOriginalTask", 
    "FinancialOriginalTask", "FNHKOriginalTask", "FTPOriginalTask", 
    "GeneeaOriginalTask", "GenesOriginalTask", "GOSalesOriginalTask", 
    "GrantsOriginalTask", "HepatitisOriginalTask", "HockeyOriginalTask", 
    "IMDbOriginalTask", "LahmanOriginalTask",
    "LegalActsOriginalTask", "MeshOriginalTask", "MondialOriginalTask", 
    "MooneyOriginalTask", "MovieLensOriginalTask", "MuskLargeOriginalTask", 
    "MuskSmallOriginalTask", "MutagenesisOriginalTask",
    "NCAAOriginalTask", "NorthwindOriginalTask", "PimaOriginalTask", 
    "PremiereLeagueOriginalTask", "RestbaseOriginalTask",
    "SakilaOriginalTask", "SalesOriginalTask", "SameGenOriginalTask", 
    "SAPOriginalTask", "SatelliteOriginalTask", "SeznamOriginalTask", 
    "SFScoresOriginalTask", "ShakespeareOriginalTask", "StatsOriginalTask",
    "StudentLoanOriginalTask", "ThrombosisOriginalTask", "ToxicologyOriginalTask",
    "TPCCOriginalTask", "TPCDOriginalTask", "TPCDSOriginalTask", "TPCHOriginalTask",
    "TriazineOriginalTask",
    "UWCSEOriginalTask", "VisualGenomeOriginalTask", "VOCOriginalTask", 
    "WalmartOriginalTask", "WebKPOriginalTask", "WorldOriginalTask"
]
# fmt: on


class AccidentsOriginalTask(CTUEntityTask):
    entity_table = "nesreca"
    target_col = "klas_nesreca"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class AdventureWorksOriginalTask(CTUEntityTask):
    entity_table = "SalesOrderHeader"
    target_col = "TotalDue"
    task_type = TaskType.REGRESSION


# TODO: remove duplicate target cols
class AirlineOriginalTask(CTUEntityTask):
    entity_table = "On_Time_On_Time_Performance_2016_1"
    target_col = "ArrDelay"
    task_type = TaskType.REGRESSION


class AtherosclerosisOriginalTask(CTUEntityTask):
    entity_table = "Entry"
    target_col = "STAV"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class BasketballMenOriginalTask(CTUEntityTask):
    entity_table = "teams"
    target_col = "rank"
    task_type = TaskType.REGRESSION


class BasketballWomenOriginalTask(CTUEntityTask):
    entity_table = "teams"
    target_col = "playoff"
    task_type = TaskType.BINARY_CLASSIFICATION


class BiodegradabilityOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    task_type = TaskType.REGRESSION


class BupaOriginalTask(CTUEntityTask):
    entity_table = "bupa"
    target_col = "arg2"
    task_type = TaskType.BINARY_CLASSIFICATION


class CarcinogenesisOriginalTask(CTUEntityTask):
    entity_table = "canc"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class CDESchoolsOriginalTask(CTUEntityTask):
    entity_table = "satscores"
    target_col = "PctGE1500"
    task_type = TaskType.REGRESSION


class ChessOriginalTask(CTUEntityTask):
    entity_table = "game"
    target_col = "game_result"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class ClassicModelsOriginalTask(CTUEntityTask):
    entity_table = "payments"
    target_col = "amount"
    task_type = TaskType.REGRESSION


class CORAOriginalTask(CTUEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CountriesOriginalTask(CTUEntityTask):
    target_col = "2012"
    target_table = "target"
    task_type = TaskType.REGRESSION


class CraftBeerOriginalTask(CTUEntityTask):
    entity_table = "breweries"
    target_col = "state"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class CreditOriginalTask(CTULinkTask):
    entity_table = "member"
    link_table = "region"
    task_type = TaskType.LINK_PREDICTION


class DallasOriginalTask(CTUEntityTask):
    entity_table = "incidents"
    target_col = "subject_statuses"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class DCGOriginalTask(CTUEntityTask):
    entity_table = "sentences"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class DiabetesOriginalTask(CTUEntityTask):
    entity_table = "paper"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class DunurOriginalTask(CTULinkTask):
    target_col = "is_dunur"
    target_table = "target"
    task_type = TaskType.LINK_PREDICTION


class EltiOriginalTask(CTULinkTask):
    target_col = "is_elti"
    target_table = "target"
    task_type = TaskType.LINK_PREDICTION


class EmployeeOriginalTask(CTUEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION


class ErgastF1OriginalTask(CTUEntityTask):
    target_col = "win"
    target_table = "target"
    task_type = TaskType.BINARY_CLASSIFICATION


class ExpendituresOriginalTask(CTUEntityTask):
    entity_table = "EXPENDITURES"
    target_col = "GIFT"
    task_type = TaskType.BINARY_CLASSIFICATION


class FinancialOriginalTask(CTUEntityTask):
    entity_table = "loan"
    target_col = "status"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class FNHKOriginalTask(CTUEntityTask):
    entity_table = "pripady"
    target_col = "Delka_hospitalizace"
    task_type = TaskType.REGRESSION


class FTPOriginalTask(CTUEntityTask):
    entity_table = "session"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class GeneeaOriginalTask(CTUEntityTask):
    entity_table = "hl_hlasovani"
    target_col = "vysledek"
    task_type = TaskType.BINARY_CLASSIFICATION


class GenesOriginalTask(CTUEntityTask):
    entity_table = "Classification"
    target_col = "Localization"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class GOSalesOriginalTask(CTUEntityTask):
    entity_table = "go_daily_sales"
    target_col = "Quantity"
    task_type = TaskType.REGRESSION


class GrantsOriginalTask(CTUEntityTask):
    entity_table = "awards"
    target_col = "award_amount"
    task_type = TaskType.REGRESSION


class HepatitisOriginalTask(CTUEntityTask):
    entity_table = "dispat"
    target_col = "Type"
    task_type = TaskType.BINARY_CLASSIFICATION


class HockeyOriginalTask(CTUEntityTask):
    entity_table = "Master"
    target_col = "shootCatch"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class IMDbOriginalTask(CTUEntityTask):
    entity_table = "actors"
    target_col = "gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class MovieLensOriginalTask(CTUEntityTask):
    entity_table = "users"
    target_col = "u_gender"
    task_type = TaskType.BINARY_CLASSIFICATION


class LahmanOriginalTask(CTUEntityTask):
    entity_table = "salaries"
    target_col = "salary"
    task_type = TaskType.REGRESSION


class LegalActsOriginalTask(CTUEntityTask):
    entity_table = "legalacts"
    target_col = "ActKind"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MeshOriginalTask(CTUEntityTask):
    entity_table = "mesh"
    target_col = "num"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class MondialOriginalTask(CTUEntityTask):
    entity_table = "country"
    target_col = "Target"
    target_table = "target"
    task_type = TaskType.BINARY_CLASSIFICATION


class MooneyOriginalTask(CTULinkTask):
    entity_table = "uncle"
    task_type = TaskType.LINK_PREDICTION


class MuskSmallOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class MuskLargeOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "class"
    task_type = TaskType.BINARY_CLASSIFICATION


class MutagenesisOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "mutagenic"
    task_type = TaskType.BINARY_CLASSIFICATION


class NCAAOriginalTask(CTUEntityTask):
    target_col = "team_id1_wins"
    target_table = "target"
    task_type = TaskType.BINARY_CLASSIFICATION


class NorthwindOriginalTask(CTUEntityTask):
    entity_table = "Orders"
    target_col = "Freight"
    task_type = TaskType.REGRESSION


class PimaOriginalTask(CTUEntityTask):
    entity_table = "pima"
    target_col = "arg2"
    task_type = TaskType.BINARY_CLASSIFICATION


class PremiereLeagueOriginalTask(CTUEntityTask):
    entity_table = "Matches"
    target_col = "ResultOfTeamHome"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class RestbaseOriginalTask(CTUEntityTask):
    entity_table = "generalinfo"
    target_col = "review"
    task_type = TaskType.REGRESSION


class SakilaOriginalTask(CTUEntityTask):
    entity_table = "payment"
    target_col = "amount"
    task_type = TaskType.REGRESSION


class SalesOriginalTask(CTUEntityTask):
    entity_table = "Sales"
    target_col = "Quantity"
    task_type = TaskType.REGRESSION


class SameGenOriginalTask(CTULinkTask):
    target_col = "target"
    target_table = "target"
    task_type = TaskType.LINK_PREDICTION


class SAPOriginalTask(CTUEntityTask):
    entity_table = "Mailings"
    target_col = "RESPONSE"
    task_type = TaskType.BINARY_CLASSIFICATION


class SatelliteOriginalTask(CTULinkTask):
    entity_table = "tm"
    link_table = "fault"
    task_type = TaskType.LINK_PREDICTION


class SeznamOriginalTask(CTUEntityTask):
    entity_table = "probehnuto"
    target_col = "kc_proklikano"
    task_type = TaskType.REGRESSION


class SFScoresOriginalTask(CTUEntityTask):
    entity_table = "inspections"
    target_col = "score"
    task_type = TaskType.REGRESSION


class ShakespeareOriginalTask(CTULinkTask):
    entity_table = "paragraphs"
    link_table = "characters"
    task_type = TaskType.LINK_PREDICTION


class StatsOriginalTask(CTUEntityTask):
    entity_table = "users"
    target_col = "Reputation"
    task_type = TaskType.REGRESSION


class StudentLoanOriginalTask(CTUEntityTask):
    entity_table = "no_payment_due"
    target_col = "bool"
    task_type = TaskType.BINARY_CLASSIFICATION


class ThrombosisOriginalTask(CTUEntityTask):
    entity_table = "Examination"
    target_col = "Thrombosis"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class ToxicologyOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "label"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCCOriginalTask(CTUEntityTask):
    entity_table = "C_Customer"
    target_col = "c_credit"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCDOriginalTask(CTUEntityTask):
    entity_table = "dss_customer"
    target_col = "c_mktsegment"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class TPCDSOriginalTask(CTUEntityTask):
    entity_table = "customer"
    target_col = "c_preferred_cust_flag"
    task_type = TaskType.BINARY_CLASSIFICATION


class TPCHOriginalTask(CTUEntityTask):
    entity_table = "customer"
    target_col = "c_acctbal"
    task_type = TaskType.REGRESSION


class TriazineOriginalTask(CTUEntityTask):
    entity_table = "molecule"
    target_col = "activity"
    task_type = TaskType.REGRESSION


class UWCSEOriginalTask(CTUEntityTask):
    entity_table = "person"
    target_col = "inPhase"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class VisualGenomeOriginalTask(CTUEntityTask):
    entity_table = "IMG_OBJ"
    target_col = "OBJ_CLASS"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class VOCOriginalTask(CTUEntityTask):
    entity_table = "voyages"
    target_col = "arrival_harbour"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class WalmartOriginalTask(CTUEntityTask):
    entity_table = "train"
    target_col = "units"
    task_type = TaskType.REGRESSION


class WebKPOriginalTask(CTUEntityTask):
    entity_table = "webpage"
    target_col = "class_label"
    task_type = TaskType.MULTICLASS_CLASSIFICATION


class WorldOriginalTask(CTUEntityTask):
    entity_table = "Country"
    target_col = "Continent"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
