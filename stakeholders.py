"""Stakeholder groups -> keywords (based on the stakeholder table).

EDIT THE LISTS to tune the analysis. Matching is case-insensitive, on whole
words, and plural -s/-es is added automatically (list irregular plurals such
as "universities" explicitly). One sentence can count for several groups.
"""

STAKEHOLDERS = {
    "Primary producers": [
        "primary producer", "agricultural producer", "farmer", "smallholder",
        "commercial farm", "farm", "grower", "rancher", "herder",
        "livestock producer", "livestock farmer", "fisher", "fisherman",
        "fishermen", "fisherfolk", "aquaculture", "fish farmer", "forester",
        "forest owner",
    ],
    "Food service": [
        "restaurant", "cafe", "café", "catering", "caterer", "street food",
        "food vendor", "street vendor", "food service", "chef", "hospitality",
    ],
    "Civil society and advocacy": [
        "ngo", "non-governmental", "non-profit", "nonprofit", "civil society",
        "advocacy", "advocate", "activist", "cooperative", "farmers union",
        "farmer union", "trade union", "labour union", "labor union",
        "consumer association", "consumer group", "charity", "food bank",
    ],
    "Input suppliers": [
        "input supplier", "agri-input", "seed company", "seed companies", "seed supplier",
        "fertilizer", "fertiliser", "pesticide", "herbicide", "agrochemical",
        "agricultural machinery", "farm machinery", "tractor", "animal feed",
        "feed supplier", "machinery provider",
    ],
    "Consumers": [
        "consumer", "household", "shopper", "institutional consumer", "school",
        "hospital", "school meal", "patient",
    ],
    "Research and education": [
        "university", "universities", "research institute", "research centre", "research center",
        "researcher", "scientist", "academic", "scholar", "extension service",
        "extension officer", "think tank", "think tanks", "think-tank",
    ],
    "Processors and manufacturers": [
        "food processor", "food processing", "processor", "manufacturer",
        "packaging", "beverage", "food industry", "food company", "food companies", "miller",
        "meat packer", "bottler", "dairy company", "dairy companies",
    ],
    "Governance and policy actors": [
        "government", "ministry", "minister", "policymaker", "policy maker",
        "regulator", "regulation", "donor", "fao", "oecd", "wto", "world bank",
        "united nations", "european commission", "environmental authority",
        "environmental agency", "food safety authority", "public health",
        "authority", "authorities", "agency", "agencies", "municipality", "parliament", "lawmaker",
    ],
    "Technology and innovation": [
        "agri-tech", "agritech", "agtech", "startup", "start-up",
        "digital platform", "e-commerce", "traceability", "blockchain",
        "biotechnology", "biotech", "precision agriculture", "drone",
    ],
    "Distributors and retailers": [
        "wholesaler", "supermarket", "hypermarket", "grocery", "grocer",
        "retailer", "retail", "distributor", "local market", "farmers market", "farmers markets",
        "convenience store", "online food retail",
    ],
    "Financial and business services": [
        "bank", "microfinance", "insurer", "insurance", "investor",
        "investment fund", "agribusiness", "lender", "financier",
        "financial institution", "venture capital", "loan",
    ],
    "Logistic and infrastructures": [
        "transport", "transporter", "trucking", "truck", "cold chain",
        "warehouse", "warehousing", "logistics", "shipping", "freight",
        "haulage", "carrier",
    ],
}
