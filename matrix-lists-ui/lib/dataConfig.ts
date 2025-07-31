export interface DataFileConfig {
  name: string
  url: string
  type: string
  fileFormat: 'tsv' | 'xlsx'
  filterColumns: string[]
  displayColumns: string[]
}

export const dataFiles: DataFileConfig[] = [
  {
    name: "Drugs List",
    url: "https://github.com/everycure-org/matrix-drug-list/releases/latest/download/drugList.tsv",
    type: "drug",
    fileFormat: 'tsv',
    filterColumns: ["curie_label", "curie", "drug_name", "ingredients_list"],
    displayColumns: ["curie_label", "curie", "drug_name", "ingredients_list", "atc_codes", "l1_label"]
  },
  {
    name: "Diseases List",
    url: "https://github.com/everycure-org/matrix-disease-list/releases/latest/download/matrix-disease-list.tsv",
    type: "disease",
    fileFormat: 'tsv',
    filterColumns: ["label", "category_class", "definition", "synonyms"],
    displayColumns: ["label", "category_class", "definition", "synonyms", "subsets"]
  },
  {
    name: "Indications List",
    url: "https://github.com/everycure-org/matrix-indication-list/releases/latest/download/indicationList.xlsx",
    type: "indication",
    fileFormat: 'xlsx',
    filterColumns: ["final normalized drug label", "final normalized disease label"],
    displayColumns: ["final normalized drug label", "final normalized disease label", "final normalized drug id", "final normalized disease id", "drug|disease"]
  },
  {
    name: "Contraindications List",
    url: "https://github.com/everycure-org/matrix-indication-list/releases/latest/download/contraindicationList.xlsx",
    type: "contraindication",
    fileFormat: 'xlsx',
    filterColumns: ["active ingredient", "contraindications", "final normalized drug label", "final normalized disease label"],
    displayColumns: ["active ingredient", "contraindications", "final normalized drug label", "final normalized disease label", "is_allergen"]
  }
] 