OffSIDES and TwoSIDES databases. Below are column definiations for the provided tables.

OFFSIDES.csv.gz/xz
==================

OffSIDES is a database of individual drug side effect signals mined from the FDA's Adverse Event Reporting System. The innovation of OffSIDES is that a propensity score matching (PSM) model is used to identify control drugs and produce better PRR estimates. In OffSIDES we focus on drug safety signals that are not already established by being listed on the structured product label -- hence they are "off-label" drug side effects. 

drug_rxnorm_id		RxNORM identifier for the drug
drug_concept_name	RxNORM name string
condition_meddra_id	MedDRA identifier for the side effect
condition_concept_name	MedDRA name string
A			The number of reports for the drug that report the side effect
B			The number of reports for the drug that do not report the side effect
C			The number of reports for other PSM matched drugs that report the side effect
D			The number of reports for other PSM matched drugs and other side effects
PRR			Proportional reporting ratio, PRR=(A/(A+B))/(C/(C+D))
PRR_error		Error estimate of the PRR
mean_reporting_frequency Proportion of reports for the drug that report the side effect, A/(A+B)


TWOSIDES.csv.gz/xz
==================

TwoSIDES is a database of drug-drug interaction safety signals mined from the FDA's Adverse Event Reporting System using the same approach as is used to generate OffSIDES. 

drug_1_rxnorn_id	RxNORM identifier for drug 1
drug_1_concept_name	RxNORM name string for drug 1
drug_2_rxnorm_id	RxNORM identifier for drug 2
drug_2_concept_name	RxNORM name string for drug 3
condition_meddra_id	MedDRA identifier for the side effect
condition_concpet_name	MedDRA name string for the side effect
A			The number of reports for the pair of drugs that report the side effect
B			The number of reports for the pair of drugs that do not report the side effect
C			The number of reports for other PSM matched drugs (including perhaps the single versions of drug 1 or drug 2) that report the side effect
D			The number of reports for other PSM matched drugs and other side effects
PRR			Proportional reporting ratio, PRR=(A/(A+B))/(C/(C+D))
PRR_error      		Error estimate of the PRR
mean_reporting_frequency Proportion of reports for the drug that report the side effect, A/(A+B)


effect_nsides-2019-11-13.sql.gz
===============================

SQL version of the above tables with some additional supporting dictionary tables.


Contact
=======

Please contact Dr. Nicholas Tatonetti for questions regarding these data.
