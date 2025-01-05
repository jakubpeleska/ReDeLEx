from relbench.base import TaskType


class CTUTask:
    pass


class Accidents(CTUTask):
    target_table = "nesreca"
    target_column = "klas_nesreca"
    task = TaskType.MULTICLASS_CLASSIFICATION


class AdventureWorks(CTUTask):
    target_table = "SalesOrderHeader"
    target_column = "TotalDue"
    task = TaskType.REGRESSION


# TODO: remove duplicate cols
class Airline(CTUTask):
    target_table = "On_Time_On_Time_Performance_2016_1"
    target_column = "ArrDelay"
    task = TaskType.REGRESSION


class Atherosclerosis(CTUTask):
    target_table = "Entry"
    target_column = "STAV"
    task = TaskType.MULTICLASS_CLASSIFICATION


class BasketballMen(CTUTask):
    target_table = "teams"
    target_column = "rank"
    task = TaskType.REGRESSION


class BasketballWomen(CTUTask):
    target_table = "teams"
    target_column = "playoff"
    task = TaskType.BINARY_CLASSIFICATION


class Biodegradability(CTUTask):
    target_table = "molecule"
    target_column = "activity"
    task = TaskType.REGRESSION


class Bookstore(CTUTask):
    target_table = "titles"
    target_column = "ytd_sales"
    task = TaskType.REGRESSION


class Bupa(CTUTask):
    target_table = "bupa"
    target_column = "arg2"
    task = TaskType.BINARY_CLASSIFICATION


class Carcinogenesis(CTUTask):
    target_table = "canc"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class CDESchools(CTUTask):
    target_table = "satscores"
    target_column = "PctGE1500"
    task = TaskType.REGRESSION


class Chess(CTUTask):
    target_table = "game"
    target_column = "game_result"
    task = TaskType.MULTICLASS_CLASSIFICATION


class ClassicModels(CTUTask):
    target_table = "payments"
    target_column = "amount"
    task = TaskType.REGRESSION


class CORA(CTUTask):
    target_table = "paper"
    target_column = "class_label"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Countries(CTUTask):
    target_table = "target"
    target_column = "2012"
    task = TaskType.REGRESSION


class CraftBeer(CTUTask):
    target_table = "breweries"
    target_column = "state"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Credit(CTUTask):
    target_table = "member"
    link_table = "region"
    task = TaskType.LINK_PREDICTION


class CS(CTUTask):
    target_table = "target_churn"
    target_column = "target_churn"
    task = TaskType.BINARY_CLASSIFICATION


class Dallas(CTUTask):
    target_table = "incidents"
    target_column = "subject_statuses"
    task = TaskType.MULTICLASS_CLASSIFICATION


class DCG(CTUTask):
    target_table = "sentences"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class Diabetes(CTUTask):
    target_table = "paper"
    target_column = "class_label"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Dunur(CTUTask):
    target_table = "target"
    target_column = "is_dunur"
    task = TaskType.BINARY_CLASSIFICATION


class Elti(CTUTask):
    target_table = "target"
    target_column = "is_elti"
    task = TaskType.BINARY_CLASSIFICATION


class Employee(CTUTask):
    target_table = "salaries"
    target_column = "salary"
    task = TaskType.REGRESSION


class ErgastF1(CTUTask):
    target_table = "target"
    target_column = "win"
    task = TaskType.BINARY_CLASSIFICATION


class Expenditures(CTUTask):
    target_table = "EXPENDITURES"
    target_column = "GIFT"
    split_column = "IS_TRAINING"
    task = TaskType.BINARY_CLASSIFICATION


class Financial(CTUTask):
    target_table = "loan"
    target_column = "status"
    task = TaskType.MULTICLASS_CLASSIFICATION


class FNHK(CTUTask):
    target_table = "pripady"
    target_column = "Delka_hospitalizace"
    task = TaskType.REGRESSION


class FTP(CTUTask):
    target_table = "session"
    target_column = "gender"
    task = TaskType.BINARY_CLASSIFICATION


class Geneea(CTUTask):
    target_table = "hl_hlasovani"
    target_column = "vysledek"
    task = TaskType.BINARY_CLASSIFICATION


class Genes(CTUTask):
    target_table = "Classification"
    target_column = "Localization"
    task = TaskType.MULTICLASS_CLASSIFICATION


class GOSales(CTUTask):
    target_table = "go_daily_sales"
    target_column = "Quantity"
    task = TaskType.REGRESSION


class Grants(CTUTask):
    target_table = "awards"
    target_column = "award_amount"
    task = TaskType.REGRESSION


class Hepatitis(CTUTask):
    target_table = "dispat"
    target_column = "Type"
    task = TaskType.BINARY_CLASSIFICATION


class Hockey(CTUTask):
    target_table = "Master"
    target_column = "shootCatch"
    task = TaskType.MULTICLASS_CLASSIFICATION


class IMDb(CTUTask):
    arget_table = "actors"
    target_column = "gender"
    task = TaskType.BINARY_CLASSIFICATION


class MovieLens(CTUTask):
    target_table = "users"
    target_column = "u_gender"
    task = TaskType.BINARY_CLASSIFICATION


class KRK(CTUTask):
    target_table = "krk"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class Lahman(CTUTask):
    target_table = "salaries"
    target_column = "salary"
    task = TaskType.REGRESSION


class LegalActs(CTUTask):
    target_table = "legalacts"
    target_column = "ActKind"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Mesh(CTUTask):
    target_table = "mesh"
    target_column = "num"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Mondial(CTUTask):
    target_table = "target"
    target_column = "Target"
    task = TaskType.BINARY_CLASSIFICATION


class Mooney(CTUTask):
    target_table = "uncle"
    task = TaskType.LINK_PREDICTION


class MuskSmall(CTUTask):
    target_table = "molecule"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class MuskLarge(CTUTask):
    target_table = "molecule"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class Mutagenesis(CTUTask):
    target_table = "molecule"
    target_column = "mutagenic"
    task = TaskType.BINARY_CLASSIFICATION


class Nations(CTUTask):
    target_table = "stat"
    target_column = "femaleworkers"
    task = TaskType.BINARY_CLASSIFICATION


class NCAA(CTUTask):
    target_table = "target"
    target_column = "team_id1_wins"
    task = TaskType.BINARY_CLASSIFICATION


class Northwind(CTUTask):
    target_table = "Orders"
    target_column = "Freight"
    task = TaskType.REGRESSION


class Pima(CTUTask):
    target_table = "pima"
    target_column = "arg2"
    task = TaskType.BINARY_CLASSIFICATION


class PremiereLeague(CTUTask):
    target_table = "Matches"
    target_column = "ResultOfTeamHome"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Pyrimidine(CTUTask):
    target_table = "molecule"
    target_column = "activity"
    task = TaskType.REGRESSION


class Restbase(CTUTask):
    target_table = "generalinfo"
    target_column = "review"
    task = TaskType.REGRESSION


class Sakila(CTUTask):
    target_table = "payment"
    target_column = "amount"
    task = TaskType.REGRESSION


class Sales(CTUTask):
    target_table = "Sales"
    target_column = "Quantity"
    task = TaskType.REGRESSION


class SameGen(CTUTask):
    target_table = "target"
    target_column = "target"
    task = TaskType.LINK_PREDICTION


class SAP(CTUTask):
    target_table = "Mailings"
    target_column = "RESPONSE"
    task = TaskType.BINARY_CLASSIFICATION


class Satellite(CTUTask):
    target_table = "tm"
    link_table = "fault"
    task = TaskType.LINK_PREDICTION


class Seznam(CTUTask):
    target_table = "probehnuto"
    target_column = "kc_proklikano"
    task = TaskType.REGRESSION


class SFScores(CTUTask):
    target_table = "inspections"
    target_column = "score"
    task = TaskType.REGRESSION


class Shakespeare(CTUTask):
    target_table = "paragraphs"
    link_table = "characters"
    task = TaskType.LINK_PREDICTION


class Stats(CTUTask):
    target_table = "users"
    target_column = "Reputation"
    task = TaskType.REGRESSION


class StudentLoan(CTUTask):
    target_table = "no_payment_due"
    target_column = "bool"
    task = TaskType.BINARY_CLASSIFICATION


class Thrombosis(CTUTask):
    target_table = "Examination"
    target_column = "Thrombosis"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Toxicology(CTUTask):
    target_table = "molecule"
    target_column = "label"
    task = TaskType.BINARY_CLASSIFICATION


class TPCC(CTUTask):
    target_table = "C_Customer"
    target_column = "c_credit"
    task = TaskType.BINARY_CLASSIFICATION


class TPCD(CTUTask):
    target_table = "dss_customer"
    target_column = "c_mktsegment"
    task = TaskType.MULTICLASS_CLASSIFICATION


class TPCDS(CTUTask):
    target_table = "customer"
    target_column = "c_preferred_cust_flag"
    task = TaskType.BINARY_CLASSIFICATION


class TPCH(CTUTask):
    target_table = "customer"
    target_column = "c_acctbal"
    task = TaskType.REGRESSION


class Triazine(CTUTask):
    target_table = "molecule"
    target_column = "activity"
    task = TaskType.REGRESSION


class University(CTUTask):
    target_table = "student"
    target_column = "intelligence"
    task = TaskType.MULTICLASS_CLASSIFICATION


class UTube(CTUTask):
    target_table = "utube_states"
    target_column = "class"
    task = TaskType.BINARY_CLASSIFICATION


class UWCSE(CTUTask):
    target_table = "person"
    target_column = "inPhase"
    task = TaskType.MULTICLASS_CLASSIFICATION


class VisualGenome(CTUTask):
    target_table = "IMG_OBJ"
    target_column = "OBJ_CLASS"
    task = TaskType.MULTICLASS_CLASSIFICATION


class VOC(CTUTask):
    target_table = "voyages"
    target_column = "arrival_harbour"
    task = TaskType.MULTICLASS_CLASSIFICATION


class Walmart(CTUTask):
    target_table = "train"
    target_column = "units"
    task = TaskType.REGRESSION


class WebKP(CTUTask):
    target_table = "webpage"
    target_column = "class_label"
    task = TaskType.MULTICLASS_CLASSIFICATION


class World(CTUTask):
    target_table = "Country"
    target_column = "Continent"
    task_type = TaskType.MULTICLASS_CLASSIFICATION
