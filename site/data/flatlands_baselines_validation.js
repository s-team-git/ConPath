window.CONPATH_FLATLANDS_BASELINES = {
  "schema_version": 1,
  "project": "ConPath",
  "dataset": "FlatLands",
  "kind": "validation_baseline_snapshot",
  "paper_result": false,
  "test_evaluated": false,
  "protocol": "P1_BASELINE_PROTOCOL.md v1",
  "primary_weighting": "equal source-scene, then equal event within scene",
  "radii_cells": [
    0,
    10,
    20
  ],
  "methods": [
    {
      "id": "radius_prior_control",
      "name": "Radius-prior control",
      "report": "results/p1_flatlands_radius_prior_validation/evaluation/report.json",
      "overall": {
        "scope": "overall",
        "source_dataset": null,
        "radius_cells": null,
        "scene_weighted": {
          "brier": 0.15869953759282368,
          "nll": 0.4928329398720715,
          "ece": 0.016610993322228283,
          "false_safe_rate@0.8": 0.11155741754333304,
          "high_confidence_safe_coverage@0.8": 0.3333333333333333,
          "positive_rate": 0.46667441565187173,
          "mean_probability": 0.46519179783662185,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.05,
              "accuracy": null
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 1408,
              "weight": 0.3333333333333333,
              "confidence": 0.17958850463020215,
              "accuracy": 0.2067289213364193
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.25,
              "accuracy": null
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 1408,
              "weight": 0.3333333333333333,
              "confidence": 0.30975822904184874,
              "accuracy": 0.30485174316252894
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.45,
              "accuracy": null
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.55,
              "accuracy": null
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.6500000000000001,
              "accuracy": null
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.75,
              "accuracy": null
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.8500000000000001,
              "accuracy": null
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 1408,
              "weight": 0.3333333333333333,
              "confidence": 0.9062286598378148,
              "accuracy": 0.8884425824566669
            }
          ]
        },
        "query_weighted": {
          "brier": 0.1864015421661864,
          "nll": 0.5453888818972241,
          "ece": 0.11861502034519629,
          "false_safe_rate@0.8": 0.04048295454545456,
          "high_confidence_safe_coverage@0.8": 0.33333333333333326,
          "positive_rate": 0.5838068181818181,
          "mean_probability": 0.4651917978366219,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.05,
              "accuracy": null
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 1408,
              "weight": 0.33333333333333326,
              "confidence": 0.17958850463020226,
              "accuracy": 0.33806818181818193
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.25,
              "accuracy": null
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 1408,
              "weight": 0.33333333333333326,
              "confidence": 0.30975822904184874,
              "accuracy": 0.45383522727272735
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.45,
              "accuracy": null
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.55,
              "accuracy": null
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.6500000000000001,
              "accuracy": null
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.75,
              "accuracy": null
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.8500000000000001,
              "accuracy": null
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 1408,
              "weight": 0.33333333333333326,
              "confidence": 0.906228659837815,
              "accuracy": 0.9595170454545456
            }
          ]
        },
        "scene_bootstrap_95": {
          "unit": "(source_dataset, scene_id)",
          "samples": 2000,
          "brier_95": [
            0.14159907327138574,
            0.17721483541628497
          ],
          "nll_95": [
            0.45122348331618545,
            0.5372290714207661
          ],
          "ece_95": [
            0.008685611091431062,
            0.053653021999066575
          ],
          "false_safe_rate@0.8_95": [
            0.07040162546764661,
            0.15938010786162193
          ]
        }
      },
      "by_radius": [
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 0,
          "scene_weighted": {
            "brier": 0.09942870468300366,
            "nll": 0.3515240802430436,
            "ece": 0.017786077381147902,
            "false_safe_rate@0.8": 0.11155741754333305,
            "high_confidence_safe_coverage@0.8": 1.0,
            "positive_rate": 0.888442582456667,
            "mean_probability": 0.9062286598378149,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.0416837369783626,
            "nll": 0.19029646622116872,
            "ece": 0.053288385616730674,
            "false_safe_rate@0.8": 0.04048295454545454,
            "high_confidence_safe_coverage@0.8": 1.0000000000000002,
            "positive_rate": 0.9595170454545456,
            "mean_probability": 0.906228659837815,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06639811475288067,
              0.1349314161845937
            ],
            "nll_95": [
              0.25930057721144206,
              0.4506499035699528
            ],
            "ece_95": [
              0.0009559312948366733,
              0.06148401774281724
            ],
            "false_safe_rate@0.8_95": [
              0.07090224818217776,
              0.15525535790500564
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 10,
          "scene_weighted": {
            "brier": 0.2119412314569804,
            "nll": 0.6149757611237081,
            "ece": 0.0049064858793198,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.3048517431625289,
            "mean_probability": 0.3097582290418487,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.26862699517825983,
            "nll": 0.7343487542326304,
            "ece": 0.14407699823087858,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.4538352272727273,
            "mean_probability": 0.3097582290418487,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.1898813419774406,
              0.23521208187931034
            ],
            "nll_95": [
              0.5685204456802618,
              0.663981205961011
            ],
            "ece_95": [
              0.0008854610761788381,
              0.0682071501883096
            ],
            "false_safe_rate@0.8_95": null
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 20,
          "scene_weighted": {
            "brier": 0.16472867663848695,
            "nll": 0.5119989782494629,
            "ece": 0.027140416706217113,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.20672892133641926,
            "mean_probability": 0.17958850463020215,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.24889389434193687,
            "nll": 0.7115214252378732,
            "ece": 0.15847967718797973,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.3380681818181819,
            "mean_probability": 0.17958850463020215,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.13202658890140156,
              0.19878857828543794
            ],
            "nll_95": [
              0.43447526067384123,
              0.5927415356409076
            ],
            "ece_95": [
              0.0013437440630515586,
              0.08029066590005632
            ],
            "false_safe_rate@0.8_95": null
          }
        }
      ],
      "by_source": [
        {
          "scope": "source",
          "source_dataset": "3RScan",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.16203941504644293,
            "nll": 0.5217860838344133,
            "ece": 0.12071246764300594,
            "false_safe_rate@0.8": 0.2555555555555556,
            "high_confidence_safe_coverage@0.8": 0.3333333333333333,
            "positive_rate": 0.3444793301936159,
            "mean_probability": 0.4651917978366219,
            "count": 417,
            "scene_count": 21
          },
          "query_weighted": {
            "brier": 0.1794638844729664,
            "nll": 0.5336645216674871,
            "ece": 0.08157079209143563,
            "false_safe_rate@0.8": 0.06474820143884893,
            "high_confidence_safe_coverage@0.8": 0.33333333333333326,
            "positive_rate": 0.5467625899280575,
            "mean_probability": 0.46519179783662185,
            "count": 417,
            "scene_count": 21
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.11379588884676892,
              0.21079787405161846
            ],
            "nll_95": [
              0.39283863859700413,
              0.6526814791130752
            ],
            "ece_95": [
              0.038576483336366256,
              0.22518045996814112
            ],
            "false_safe_rate@0.8_95": [
              0.09833333333333336,
              0.42857142857142866
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ARKitScenes",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.12123888017072877,
            "nll": 0.42556966836758475,
            "ece": 0.17877798017280416,
            "false_safe_rate@0.8": 0.22970085470085475,
            "high_confidence_safe_coverage@0.8": 0.3333333333333333,
            "positive_rate": 0.2864138176638177,
            "mean_probability": 0.4651917978366218,
            "count": 381,
            "scene_count": 26
          },
          "query_weighted": {
            "brier": 0.1124466953535111,
            "nll": 0.3894104746077734,
            "ece": 0.09773772959515192,
            "false_safe_rate@0.8": 0.11811023622047241,
            "high_confidence_safe_coverage@0.8": 0.3333333333333334,
            "positive_rate": 0.3674540682414698,
            "mean_probability": 0.4651917978366219,
            "count": 381,
            "scene_count": 26
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.08677612938273159,
              0.1608768270458175
            ],
            "nll_95": [
              0.3327978060765748,
              0.5352016744360032
            ],
            "ece_95": [
              0.11770105663989483,
              0.2408154623665364
            ],
            "false_safe_rate@0.8_95": [
              0.1061858974358974,
              0.37735042735042723
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "Matterport3D",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.189353689086373,
            "nll": 0.5577694329331245,
            "ece": 0.08945471816388399,
            "false_safe_rate@0.8": 0.08128027886092402,
            "high_confidence_safe_coverage@0.8": 0.33333333333333337,
            "positive_rate": 0.5546465160005059,
            "mean_probability": 0.4651917978366219,
            "count": 1104,
            "scene_count": 31
          },
          "query_weighted": {
            "brier": 0.2141567227366819,
            "nll": 0.6099954276146584,
            "ece": 0.16343139056917522,
            "false_safe_rate@0.8": 0.05163043478260869,
            "high_confidence_safe_coverage@0.8": 0.33333333333333337,
            "positive_rate": 0.6286231884057971,
            "mean_probability": 0.46519179783662185,
            "count": 1104,
            "scene_count": 31
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.15409815573355673,
              0.22351569618395364
            ],
            "nll_95": [
              0.4770302112282938,
              0.6386004723741087
            ],
            "ece_95": [
              0.024921147526268582,
              0.17744943166581303
            ],
            "false_safe_rate@0.8_95": [
              0.026362548741580987,
              0.1472446236559139
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ScanNet",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.11002544446009477,
            "nll": 0.3749343633097444,
            "ece": 0.07089197639111379,
            "false_safe_rate@0.8": 0.06195549242424242,
            "high_confidence_safe_coverage@0.8": 0.3333333333333333,
            "positive_rate": 0.4155103866041366,
            "mean_probability": 0.4651917978366219,
            "count": 1008,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.12282700475191317,
            "nll": 0.40085833603807214,
            "ece": 0.04060285000823739,
            "false_safe_rate@0.8": 0.041666666666666664,
            "high_confidence_safe_coverage@0.8": 0.3333333333333333,
            "positive_rate": 0.45932539682539675,
            "mean_probability": 0.4651917978366219,
            "count": 1008,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.08043686721973936,
              0.14360048739647208
            ],
            "nll_95": [
              0.3029096700569727,
              0.4588126400839476
            ],
            "ece_95": [
              0.022626482725335455,
              0.12829156886387955
            ],
            "false_safe_rate@0.8_95": [
              0.00596590909090909,
              0.14678740530303025
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ZInD",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.2059224110426911,
            "nll": 0.5834751961536121,
            "ece": 0.19407593534752005,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.33333333333333337,
            "positive_rate": 0.6592677331841419,
            "mean_probability": 0.46519179783662185,
            "count": 1314,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.2354968345995673,
            "nll": 0.6509276909893538,
            "ece": 0.2509421443247176,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.3333333333333333,
            "positive_rate": 0.7161339421613394,
            "mean_probability": 0.4651917978366219,
            "count": 1314,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.16752951967848503,
              0.24431905735504966
            ],
            "nll_95": [
              0.4952619968325406,
              0.6712228031543935
            ],
            "ece_95": [
              0.11863310090942633,
              0.26733076754461105
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.0
            ]
          }
        }
      ],
      "radius_monotonicity": {
        "query_count": 1408,
        "violating_queries": 0,
        "violation_rate": 0.0,
        "maximum_probability_increase": 0.0,
        "rule": "predicted reachability must be non-increasing as footprint radius grows"
      }
    },
    {
      "id": "deterministic_completion",
      "name": "Deterministic completion",
      "report": "results/p1_flatlands_completion_seed20260831/evaluation_deterministic_validation/report.json",
      "overall": {
        "scope": "overall",
        "source_dataset": null,
        "radius_cells": null,
        "scene_weighted": {
          "brier": 0.08556390714661546,
          "nll": 1.182109977000236,
          "ece": 0.08556390714661549,
          "false_safe_rate@0.8": 0.08073445916155293,
          "high_confidence_safe_coverage@0.8": 0.4544977721554598,
          "positive_rate": 0.46667441565187173,
          "mean_probability": 0.4544977721554597,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 1755,
              "weight": 0.5455022278445402,
              "confidence": 0.0,
              "accuracy": 0.08958767320642547
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.15000000000000002,
              "accuracy": null
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.25,
              "accuracy": null
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.35000000000000003,
              "accuracy": null
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.45,
              "accuracy": null
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.55,
              "accuracy": null
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.6500000000000001,
              "accuracy": null
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.75,
              "accuracy": null
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.8500000000000001,
              "accuracy": null
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 2469,
              "weight": 0.4544977721554598,
              "confidence": 1.0,
              "accuracy": 0.919265540838447
            }
          ]
        },
        "query_weighted": {
          "brier": 0.07267992424242424,
          "nll": 1.00411118804274,
          "ece": 0.07267992424242417,
          "false_safe_rate@0.8": 0.06277845281490482,
          "high_confidence_safe_coverage@0.8": 0.5845170454545454,
          "positive_rate": 0.5838068181818181,
          "mean_probability": 0.5845170454545454,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 1755,
              "weight": 0.41548295454545436,
              "confidence": 0.0,
              "accuracy": 0.08660968660968665
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.15000000000000002,
              "accuracy": null
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.25,
              "accuracy": null
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.35000000000000003,
              "accuracy": null
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.45,
              "accuracy": null
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.55,
              "accuracy": null
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.6500000000000001,
              "accuracy": null
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.75,
              "accuracy": null
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 0,
              "weight": 0.0,
              "confidence": 0.8500000000000001,
              "accuracy": null
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 2469,
              "weight": 0.5845170454545454,
              "confidence": 1.0,
              "accuracy": 0.9372215471850953
            }
          ]
        },
        "scene_bootstrap_95": {
          "unit": "(source_dataset, scene_id)",
          "samples": 2000,
          "brier_95": [
            0.06673140391700981,
            0.10489240438817056
          ],
          "nll_95": [
            0.9219293486312976,
            1.4491430153819211
          ],
          "ece_95": [
            0.06673140391700974,
            0.10489240438817057
          ],
          "false_safe_rate@0.8_95": [
            0.05473643052689216,
            0.10907791076131586
          ]
        }
      },
      "by_radius": [
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 0,
          "scene_weighted": {
            "brier": 0.13874264011375378,
            "nll": 1.9168012705887056,
            "ece": 0.13874264011375367,
            "false_safe_rate@0.8": 0.021649605987661934,
            "high_confidence_safe_coverage@0.8": 0.7836305266256101,
            "positive_rate": 0.888442582456667,
            "mean_probability": 0.7836305266256101,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.09090909090909091,
            "nll": 1.2559564143604347,
            "ece": 0.09090909090909084,
            "false_safe_rate@0.8": 0.015835312747426757,
            "high_confidence_safe_coverage@0.8": 0.8970170454545456,
            "positive_rate": 0.9595170454545456,
            "mean_probability": 0.8970170454545456,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.0956601861462556,
              0.1845632286519531
            ],
            "nll_95": [
              1.32159521602033,
              2.5498360494894117
            ],
            "ece_95": [
              0.09566018614625568,
              0.1845632286519531
            ],
            "false_safe_rate@0.8_95": [
              0.006055678999878714,
              0.04198076755158818
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 10,
          "scene_weighted": {
            "brier": 0.0819753433495094,
            "nll": 1.132532139561186,
            "ece": 0.0819753433495094,
            "false_safe_rate@0.8": 0.1813176872377018,
            "high_confidence_safe_coverage@0.8": 0.34968429512318233,
            "positive_rate": 0.3048517431625289,
            "mean_probability": 0.34968429512318233,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.08451704545454546,
            "nll": 1.167647049286814,
            "ece": 0.08451704545454544,
            "false_safe_rate@0.8": 0.1275071633237822,
            "high_confidence_safe_coverage@0.8": 0.49573863636363646,
            "positive_rate": 0.4538352272727273,
            "mean_probability": 0.49573863636363646,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.05598619831290818,
              0.11170165673827837
            ],
            "nll_95": [
              0.7734788579054379,
              1.543216306306001
            ],
            "ece_95": [
              0.05598619831290822,
              0.11170165673827842
            ],
            "false_safe_rate@0.8_95": [
              0.11727004296450094,
              0.25567594462457494
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 20,
          "scene_weighted": {
            "brier": 0.03597373797658322,
            "nll": 0.4969965208508156,
            "ece": 0.03597373797658321,
            "false_safe_rate@0.8": 0.12908093658067216,
            "high_confidence_safe_coverage@0.8": 0.23017849471758683,
            "positive_rate": 0.20672892133641926,
            "mean_probability": 0.23017849471758683,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.04261363636363636,
            "nll": 0.5887301004809714,
            "ece": 0.04261363636363634,
            "false_safe_rate@0.8": 0.09055118110236221,
            "high_confidence_safe_coverage@0.8": 0.36079545454545453,
            "positive_rate": 0.3380681818181819,
            "mean_probability": 0.3607954545454546,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.020267103292110924,
              0.05481552297138047
            ],
            "nll_95": [
              0.28000135924434766,
              0.7573053815354157
            ],
            "ece_95": [
              0.02026710329211092,
              0.05481552297138047
            ],
            "false_safe_rate@0.8_95": [
              0.07252124668174932,
              0.19747719006991965
            ]
          }
        }
      ],
      "by_source": [
        {
          "scope": "source",
          "source_dataset": "3RScan",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.09463050177335891,
            "nll": 1.3073696017245273,
            "ece": 0.09463050177335892,
            "false_safe_rate@0.8": 0.08352474825303598,
            "high_confidence_safe_coverage@0.8": 0.2999563928135357,
            "positive_rate": 0.3444793301936159,
            "mean_probability": 0.29995639281353564,
            "count": 417,
            "scene_count": 21
          },
          "query_weighted": {
            "brier": 0.05755395683453237,
            "nll": 0.7951382407457198,
            "ece": 0.0575539568345323,
            "false_safe_rate@0.8": 0.056521739130434796,
            "high_confidence_safe_coverage@0.8": 0.5515587529976017,
            "positive_rate": 0.5467625899280575,
            "mean_probability": 0.5515587529976018,
            "count": 417,
            "scene_count": 21
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.044437830687830686,
              0.14916092505378203
            ],
            "nll_95": [
              0.6139322746028143,
              2.060735185754457
            ],
            "ece_95": [
              0.04443783068783067,
              0.1491609250537819
            ],
            "false_safe_rate@0.8_95": [
              0.015503875968992248,
              0.17950739997183743
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ARKitScenes",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.1127035002035002,
            "nll": 1.5570572842771755,
            "ece": 0.1127035002035002,
            "false_safe_rate@0.8": 0.11593534526393935,
            "high_confidence_safe_coverage@0.8": 0.2261472323972324,
            "positive_rate": 0.2864138176638177,
            "mean_probability": 0.2261472323972324,
            "count": 381,
            "scene_count": 26
          },
          "query_weighted": {
            "brier": 0.14698162729658792,
            "nll": 2.0306270767601458,
            "ece": 0.1469816272965879,
            "false_safe_rate@0.8": 0.15573770491803277,
            "high_confidence_safe_coverage@0.8": 0.32020997375328086,
            "positive_rate": 0.3674540682414698,
            "mean_probability": 0.32020997375328086,
            "count": 381,
            "scene_count": 26
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06858440170940171,
              0.1628823260073259
            ],
            "nll_95": [
              0.9475294573431653,
              2.250303331776877
            ],
            "ece_95": [
              0.06858440170940172,
              0.1628823260073259
            ],
            "false_safe_rate@0.8_95": [
              0.018932473617686785,
              0.21797944608508113
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "Matterport3D",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.10145049096662001,
            "nll": 1.4015912276082032,
            "ece": 0.10145049096661996,
            "false_safe_rate@0.8": 0.10599670877559582,
            "high_confidence_safe_coverage@0.8": 0.575117055019432,
            "positive_rate": 0.5546465160005059,
            "mean_probability": 0.575117055019432,
            "count": 1104,
            "scene_count": 31
          },
          "query_weighted": {
            "brier": 0.0733695652173913,
            "nll": 1.0136389295237138,
            "ece": 0.07336956521739142,
            "false_safe_rate@0.8": 0.07012622720897616,
            "high_confidence_safe_coverage@0.8": 0.6458333333333335,
            "positive_rate": 0.6286231884057971,
            "mean_probability": 0.6458333333333333,
            "count": 1104,
            "scene_count": 31
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06135101644174224,
              0.1444416254295286
            ],
            "nll_95": [
              0.8475965540412475,
              1.9955356566870275
            ],
            "ece_95": [
              0.06135101644174228,
              0.1444416254295285
            ],
            "false_safe_rate@0.8_95": [
              0.04289706921810763,
              0.18270840694791365
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ScanNet",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.06690069252714376,
            "nll": 0.924268157042712,
            "ece": 0.06690069252714376,
            "false_safe_rate@0.8": 0.0788336877491107,
            "high_confidence_safe_coverage@0.8": 0.4138622723810417,
            "positive_rate": 0.4155103866041366,
            "mean_probability": 0.4138622723810417,
            "count": 1008,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.06746031746031746,
            "nll": 0.9319996606557879,
            "ece": 0.06746031746031741,
            "false_safe_rate@0.8": 0.07526881720430108,
            "high_confidence_safe_coverage@0.8": 0.4613095238095238,
            "positive_rate": 0.45932539682539675,
            "mean_probability": 0.4613095238095238,
            "count": 1008,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.03762273144239088,
              0.10282616027288771
            ],
            "nll_95": [
              0.5197782058391053,
              1.420596800057673
            ],
            "ece_95": [
              0.03762273144239084,
              0.10282616027288771
            ],
            "false_safe_rate@0.8_95": [
              0.03259773827208308,
              0.1368594985732852
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ZInD",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.060836121607938554,
            "nll": 0.840483019543461,
            "ece": 0.06083612160793863,
            "false_safe_rate@0.8": 0.05021100054187465,
            "high_confidence_safe_coverage@0.8": 0.6652359354021022,
            "positive_rate": 0.6592677331841419,
            "mean_probability": 0.6652359354021021,
            "count": 1314,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.0593607305936073,
            "nll": 0.8200997408833626,
            "ece": 0.059360730593607254,
            "false_safe_rate@0.8": 0.04046858359957402,
            "high_confidence_safe_coverage@0.8": 0.7146118721461187,
            "positive_rate": 0.7161339421613394,
            "mean_probability": 0.7146118721461187,
            "count": 1314,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.028717548231746177,
              0.10557538572936817
            ],
            "nll_95": [
              0.3967485620769206,
              1.4585787506289756
            ],
            "ece_95": [
              0.028717548231746166,
              0.10557538572936795
            ],
            "false_safe_rate@0.8_95": [
              0.023468406530545993,
              0.0839463308051558
            ]
          }
        }
      ],
      "radius_monotonicity": {
        "query_count": 1408,
        "violating_queries": 0,
        "violation_rate": 0.0,
        "maximum_probability_increase": 0.0,
        "rule": "predicted reachability must be non-increasing as footprint radius grows"
      }
    },
    {
      "id": "independent_cell_completion_k32",
      "name": "Independent-cell completion (K=32)",
      "report": "results/p1_flatlands_completion_seed20260831/evaluation_independent_k32_validation/report.json",
      "overall": {
        "scope": "overall",
        "source_dataset": null,
        "radius_cells": null,
        "scene_weighted": {
          "brier": 0.22545963439115663,
          "nll": 2.9276834171557624,
          "ece": 0.24024873736506533,
          "false_safe_rate@0.8": 0.014543295956701964,
          "high_confidence_safe_coverage@0.8": 0.20624474400583062,
          "positive_rate": 0.46667441565187173,
          "mean_probability": 0.22642567828680635,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 2957,
              "weight": 0.7374768540430467,
              "confidence": 0.0002937783830408739,
              "accuracy": 0.2839289850797168
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 14,
              "weight": 0.0029175520607082114,
              "confidence": 0.14462013020587797,
              "accuracy": 1.0
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 14,
              "weight": 0.003038562655708253,
              "confidence": 0.253800655489478,
              "accuracy": 1.0
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 17,
              "weight": 0.004462717734258745,
              "confidence": 0.3447794222166183,
              "accuracy": 0.9473993655848931
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 15,
              "weight": 0.0038609189914078066,
              "confidence": 0.4318696813301383,
              "accuracy": 0.9532311857597474
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 37,
              "weight": 0.010377853887207224,
              "confidence": 0.5388196101275059,
              "accuracy": 0.936926416253994
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 61,
              "weight": 0.013077760012581013,
              "confidence": 0.6559876234582925,
              "accuracy": 0.954435393394898
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 102,
              "weight": 0.01854303660925124,
              "confidence": 0.7555503697071609,
              "accuracy": 0.9690157097690617
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 241,
              "weight": 0.04437953027703606,
              "confidence": 0.848752489089026,
              "accuracy": 0.9853071786216526
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 766,
              "weight": 0.1618652137287946,
              "confidence": 0.9635285558775043,
              "accuracy": 0.9854977003007657
            }
          ]
        },
        "query_weighted": {
          "brier": 0.3037053888494319,
          "nll": 4.008577910579021,
          "ece": 0.32442589962121193,
          "false_safe_rate@0.8": 0.006951340615690171,
          "high_confidence_safe_coverage@0.8": 0.23839962121212113,
          "positive_rate": 0.5838068181818181,
          "mean_probability": 0.2593809185606061,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 2957,
              "weight": 0.7000473484848483,
              "confidence": 0.0003276124450456545,
              "accuracy": 0.41156577612445056
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 14,
              "weight": 0.003314393939393938,
              "confidence": 0.1495535714285715,
              "accuracy": 1.0
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 14,
              "weight": 0.003314393939393938,
              "confidence": 0.2544642857142857,
              "accuracy": 1.0
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 17,
              "weight": 0.004024621212121212,
              "confidence": 0.3455882352941177,
              "accuracy": 0.9411764705882354
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 15,
              "weight": 0.0035511363636363622,
              "confidence": 0.4354166666666669,
              "accuracy": 0.9333333333333333
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 37,
              "weight": 0.008759469696969696,
              "confidence": 0.5396959459459459,
              "accuracy": 0.918918918918919
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 61,
              "weight": 0.014441287878787878,
              "confidence": 0.6572745901639344,
              "accuracy": 0.9508196721311475
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 102,
              "weight": 0.024147727272727265,
              "confidence": 0.7552083333333336,
              "accuracy": 0.9705882352941176
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 241,
              "weight": 0.05705492424242422,
              "confidence": 0.849066390041494,
              "accuracy": 0.991701244813278
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 766,
              "weight": 0.1813446969696969,
              "confidence": 0.959366840731071,
              "accuracy": 0.993472584856397
            }
          ]
        },
        "scene_bootstrap_95": {
          "unit": "(source_dataset, scene_id)",
          "samples": 2000,
          "brier_95": [
            0.19174074369959787,
            0.25968376316548075
          ],
          "nll_95": [
            2.4598958788869636,
            3.4052366952667144
          ],
          "ece_95": [
            0.20411880658148907,
            0.27623434237607897
          ],
          "false_safe_rate@0.8_95": [
            0.0,
            0.036208617797997084
          ]
        }
      },
      "by_radius": [
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 0,
          "scene_weighted": {
            "brier": 0.16479823867452165,
            "nll": 1.7153006914116107,
            "ece": 0.2091655475962479,
            "false_safe_rate@0.8": 0.01454329595670196,
            "high_confidence_safe_coverage@0.8": 0.6187342320174919,
            "positive_rate": 0.888442582456667,
            "mean_probability": 0.6792770348604191,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.11921275745738637,
            "nll": 1.085182614456511,
            "ece": 0.1813742897727274,
            "false_safe_rate@0.8": 0.006951340615690169,
            "high_confidence_safe_coverage@0.8": 0.7151988636363635,
            "positive_rate": 0.9595170454545456,
            "mean_probability": 0.7781427556818181,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.12280500236607295,
              0.21029358431848547
            ],
            "nll_95": [
              1.1641779791429532,
              2.326278664783185
            ],
            "ece_95": [
              0.1637567925241543,
              0.2578813329744532
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.03612823690270571
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 10,
          "scene_weighted": {
            "brier": 0.3048517431625289,
            "nll": 4.211683171424335,
            "ece": 0.3048517431625289,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.3048517431625289,
            "mean_probability": 0.0,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.4538352272727273,
            "nll": 6.269965920127525,
            "ece": 0.4538352272727273,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.4538352272727273,
            "mean_probability": 0.0,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.2468731789165294,
              0.3660129968270802
            ],
            "nll_95": [
              3.4106797629267125,
              5.056657056003993
            ],
            "ece_95": [
              0.24687317891652946,
              0.3660129968270803
            ],
            "false_safe_rate@0.8_95": null
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 20,
          "scene_weighted": {
            "brier": 0.20672892133641926,
            "nll": 2.8560663886313415,
            "ece": 0.20672892133641926,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.20672892133641926,
            "mean_probability": 0.0,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.3380681818181819,
            "nll": 4.670585197153026,
            "ece": 0.3380681818181819,
            "false_safe_rate@0.8": null,
            "high_confidence_safe_coverage@0.8": 0.0,
            "positive_rate": 0.3380681818181819,
            "mean_probability": 0.0,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.15569753168645864,
              0.2598791705302587
            ],
            "nll_95": [
              2.151041737166138,
              3.5903641643769877
            ],
            "ece_95": [
              0.15569753168645864,
              0.2598791705302587
            ],
            "false_safe_rate@0.8_95": null
          }
        }
      ],
      "by_source": [
        {
          "scope": "source",
          "source_dataset": "3RScan",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.16804391561646026,
            "nll": 2.270766692755367,
            "ece": 0.17961109657538227,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.1623466480609338,
            "positive_rate": 0.3444793301936159,
            "mean_probability": 0.1648682336182336,
            "count": 417,
            "scene_count": 21
          },
          "query_weighted": {
            "brier": 0.2658357688848921,
            "nll": 3.6114764800453707,
            "ece": 0.2855215827338129,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.25899280575539557,
            "positive_rate": 0.5467625899280575,
            "mean_probability": 0.2612410071942446,
            "count": 417,
            "scene_count": 21
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.09510597327038454,
              0.2466914874882578
            ],
            "nll_95": [
              1.2489926902761315,
              3.3516059494555184
            ],
            "ece_95": [
              0.10508506353930462,
              0.2591620106910285
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.0
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ARKitScenes",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.128878827652021,
            "nll": 1.5315070338432961,
            "ece": 0.1441171747812373,
            "false_safe_rate@0.8": 0.013014781072789814,
            "high_confidence_safe_coverage@0.8": 0.10945258445258443,
            "positive_rate": 0.2864138176638177,
            "mean_probability": 0.14229664288258037,
            "count": 381,
            "scene_count": 26
          },
          "query_weighted": {
            "brier": 0.18518546998031493,
            "nll": 2.1818142578315243,
            "ece": 0.20661089238845146,
            "false_safe_rate@0.8": 0.022727272727272728,
            "high_confidence_safe_coverage@0.8": 0.11548556430446194,
            "positive_rate": 0.3674540682414698,
            "mean_probability": 0.16084317585301838,
            "count": 381,
            "scene_count": 26
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.07827657237691418,
              0.18248809510146788
            ],
            "nll_95": [
              0.8293459558572274,
              2.275506967875137
            ],
            "ece_95": [
              0.09158452730718356,
              0.19981851120522987
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.039787209600311055
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "Matterport3D",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.30338977973497805,
            "nll": 3.9744900842829614,
            "ece": 0.30352577738573605,
            "false_safe_rate@0.8": 0.05168417675811426,
            "high_confidence_safe_coverage@0.8": 0.24272037699457055,
            "positive_rate": 0.5546465160005059,
            "mean_probability": 0.2596699397202103,
            "count": 1104,
            "scene_count": 31
          },
          "query_weighted": {
            "brier": 0.3514581210371377,
            "nll": 4.671372324046987,
            "ece": 0.35750679347826086,
            "false_safe_rate@0.8": 0.021126760563380278,
            "high_confidence_safe_coverage@0.8": 0.25724637681159424,
            "positive_rate": 0.6286231884057971,
            "mean_probability": 0.27111639492753625,
            "count": 1104,
            "scene_count": 31
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.24113070425634725,
              0.36486130238951264
            ],
            "nll_95": [
              3.062407682608951,
              4.8826821344026
            ],
            "ece_95": [
              0.23954911180749136,
              0.373600249905627
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.12607990058223767
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ScanNet",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.14735624549084317,
            "nll": 1.742664785471826,
            "ece": 0.17658186785988733,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.2062579519114929,
            "positive_rate": 0.4155103866041366,
            "mean_probability": 0.23892851874424922,
            "count": 1008,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.18297564794146826,
            "nll": 2.2445437092849008,
            "ece": 0.21391369047619044,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.2093253968253968,
            "positive_rate": 0.45932539682539675,
            "mean_probability": 0.24541170634920634,
            "count": 1008,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.08991983189413107,
              0.2103595085188863
            ],
            "nll_95": [
              0.9621853908872501,
              2.6342940229031084
            ],
            "ece_95": [
              0.11613901274321879,
              0.2401566471956522
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.0
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ZInD",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.34421916591108526,
            "nll": 4.664103001889364,
            "ece": 0.368798359288704,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.2783475217431777,
            "positive_rate": 0.6592677331841419,
            "mean_probability": 0.2904693738954378,
            "count": 1314,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.4025823166381278,
            "nll": 5.460640080035006,
            "ece": 0.4279157153729072,
            "false_safe_rate@0.8": 0.0,
            "high_confidence_safe_coverage@0.8": 0.273972602739726,
            "positive_rate": 0.7161339421613394,
            "mean_probability": 0.28821822678843223,
            "count": 1314,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.2654140465965229,
              0.42317624943345955
            ],
            "nll_95": [
              3.5893751656287196,
              5.730872081292064
            ],
            "ece_95": [
              0.2897045105502207,
              0.44945122543566957
            ],
            "false_safe_rate@0.8_95": [
              0.0,
              0.0
            ]
          }
        }
      ],
      "radius_monotonicity": {
        "query_count": 1408,
        "violating_queries": 0,
        "violation_rate": 0.0,
        "maximum_probability_increase": 0.0,
        "rule": "predicted reachability must be non-increasing as footprint radius grows"
      }
    },
    {
      "id": "direct_query",
      "name": "Direct-query predictor",
      "report": "results/p1_flatlands_direct_query_seed20260831/evaluation_validation/report.json",
      "overall": {
        "scope": "overall",
        "source_dataset": null,
        "radius_cells": null,
        "scene_weighted": {
          "brier": 0.09118529284559757,
          "nll": 0.2978753355577147,
          "ece": 0.04075985198992008,
          "false_safe_rate@0.8": 0.08335053174946491,
          "high_confidence_safe_coverage@0.8": 0.3741174171114105,
          "positive_rate": 0.46667441565187173,
          "mean_probability": 0.45415940984431163,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 1239,
              "weight": 0.3992885421010285,
              "confidence": 0.014827250683682203,
              "accuracy": 0.023843952405748853
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 263,
              "weight": 0.057860357365437376,
              "confidence": 0.14891117766420808,
              "accuracy": 0.29611051061901916
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 138,
              "weight": 0.026193506120085724,
              "confidence": 0.24630157663730723,
              "accuracy": 0.42305484351215566
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 107,
              "weight": 0.026339569011700873,
              "confidence": 0.33931544211224896,
              "accuracy": 0.4040836253263469
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 116,
              "weight": 0.02483408028337824,
              "confidence": 0.4436034767278671,
              "accuracy": 0.7238133985647973
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 99,
              "weight": 0.024227889510451756,
              "confidence": 0.5480990258148831,
              "accuracy": 0.5986873240679743
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 103,
              "weight": 0.022857634864622615,
              "confidence": 0.6470476182494649,
              "accuracy": 0.5889603774437705
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 166,
              "weight": 0.04428100363188446,
              "confidence": 0.7652989822482544,
              "accuracy": 0.6643733205585026
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 423,
              "weight": 0.097473961488155,
              "confidence": 0.8594420084551873,
              "accuracy": 0.7976029845682595
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 1570,
              "weight": 0.2766434556232555,
              "confidence": 0.9669012779682925,
              "accuracy": 0.9585949114911989
            }
          ]
        },
        "query_weighted": {
          "brier": 0.0894262765048364,
          "nll": 0.30430950477091984,
          "ece": 0.039893803867474036,
          "false_safe_rate@0.8": 0.055694932262920244,
          "high_confidence_safe_coverage@0.8": 0.4718276515151513,
          "positive_rate": 0.5838068181818181,
          "mean_probability": 0.5495425076612899,
          "count": 4224,
          "scene_count": 142,
          "reliability": [
            {
              "bin": 0,
              "lower": 0.0,
              "upper": 0.1,
              "count": 1239,
              "weight": 0.2933238636363636,
              "confidence": 0.0189016898608596,
              "accuracy": 0.04196933010492333
            },
            {
              "bin": 1,
              "lower": 0.1,
              "upper": 0.2,
              "count": 263,
              "weight": 0.06226325757575756,
              "confidence": 0.1463825152230807,
              "accuracy": 0.22433460076045633
            },
            {
              "bin": 2,
              "lower": 0.2,
              "upper": 0.30000000000000004,
              "count": 138,
              "weight": 0.032670454545454544,
              "confidence": 0.24707190271305005,
              "accuracy": 0.4710144927536232
            },
            {
              "bin": 3,
              "lower": 0.30000000000000004,
              "upper": 0.4,
              "count": 107,
              "weight": 0.025331439393939385,
              "confidence": 0.3454718291759492,
              "accuracy": 0.560747663551402
            },
            {
              "bin": 4,
              "lower": 0.4,
              "upper": 0.5,
              "count": 116,
              "weight": 0.0274621212121212,
              "confidence": 0.44827706264010814,
              "accuracy": 0.7413793103448277
            },
            {
              "bin": 5,
              "lower": 0.5,
              "upper": 0.6000000000000001,
              "count": 99,
              "weight": 0.023437499999999993,
              "confidence": 0.5503503776559927,
              "accuracy": 0.7272727272727274
            },
            {
              "bin": 6,
              "lower": 0.6000000000000001,
              "upper": 0.7000000000000001,
              "count": 103,
              "weight": 0.02438446969696969,
              "confidence": 0.649302048590577,
              "accuracy": 0.6407766990291264
            },
            {
              "bin": 7,
              "lower": 0.7000000000000001,
              "upper": 0.8,
              "count": 166,
              "weight": 0.03929924242424242,
              "confidence": 0.763122859489487,
              "accuracy": 0.7469879518072291
            },
            {
              "bin": 8,
              "lower": 0.8,
              "upper": 0.9,
              "count": 423,
              "weight": 0.10014204545454541,
              "confidence": 0.8613072914152847,
              "accuracy": 0.8416075650118204
            },
            {
              "bin": 9,
              "lower": 0.9,
              "upper": 1.0,
              "count": 1570,
              "weight": 0.37168560606060597,
              "confidence": 0.9706458803954399,
              "accuracy": 0.9719745222929935
            }
          ]
        },
        "scene_bootstrap_95": {
          "unit": "(source_dataset, scene_id)",
          "samples": 2000,
          "brier_95": [
            0.07869141966411602,
            0.10544156035056466
          ],
          "nll_95": [
            0.24985876619068453,
            0.35941655561897873
          ],
          "ece_95": [
            0.029392623718050067,
            0.06284282846539799
          ],
          "false_safe_rate@0.8_95": [
            0.054445593919694564,
            0.11618084748038625
          ]
        }
      },
      "by_radius": [
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 0,
          "scene_weighted": {
            "brier": 0.0950817408062625,
            "nll": 0.3024931592452895,
            "ece": 0.051073275687333754,
            "false_safe_rate@0.8": 0.05610331545568995,
            "high_confidence_safe_coverage@0.8": 0.80851254647449,
            "positive_rate": 0.888442582456667,
            "mean_probability": 0.868816733830278,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.04117161446689359,
            "nll": 0.1505059392753474,
            "ece": 0.027122958253709337,
            "false_safe_rate@0.8": 0.02911877394636015,
            "high_confidence_safe_coverage@0.8": 0.926846590909091,
            "positive_rate": 0.9595170454545456,
            "mean_probability": 0.9323940872008363,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06836195065310845,
              0.12588528941177166
            ],
            "nll_95": [
              0.22948306510828406,
              0.3869208848318505
            ],
            "ece_95": [
              0.03839799221333271,
              0.09841067912220848
            ],
            "false_safe_rate@0.8_95": [
              0.02910727143972887,
              0.0873765586326559
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 10,
          "scene_weighted": {
            "brier": 0.09056283893975746,
            "nll": 0.3004434854974232,
            "ece": 0.05670220548812412,
            "false_safe_rate@0.8": 0.15264958259318354,
            "high_confidence_safe_coverage@0.8": 0.26912118735637247,
            "positive_rate": 0.3048517431625289,
            "mean_probability": 0.3598945735249868,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.09954529240429306,
            "nll": 0.334294494684336,
            "ece": 0.0548942047440423,
            "false_safe_rate@0.8": 0.10820244328097728,
            "high_confidence_safe_coverage@0.8": 0.40696022727272735,
            "positive_rate": 0.4538352272727273,
            "mean_probability": 0.5010798498930689,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06731992298462902,
              0.11680073555330582
            ],
            "nll_95": [
              0.22644240472363192,
              0.3877195790279923
            ],
            "ece_95": [
              0.03746772148705769,
              0.09501991592120167
            ],
            "false_safe_rate@0.8_95": [
              0.084410671182472,
              0.22913747537176712
            ]
          }
        },
        {
          "scope": "radius",
          "source_dataset": null,
          "radius_cells": 20,
          "scene_weighted": {
            "brier": 0.08791129879077277,
            "nll": 0.2906893619304315,
            "ece": 0.07872361865915169,
            "false_safe_rate@0.8": 0.15893160135973086,
            "high_confidence_safe_coverage@0.8": 0.044718517503369026,
            "positive_rate": 0.20672892133641926,
            "mean_probability": 0.13376692217767017,
            "count": 1408,
            "scene_count": 142
          },
          "query_weighted": {
            "brier": 0.1275619226433225,
            "nll": 0.4281280803530762,
            "ece": 0.12483945757405118,
            "false_safe_rate@0.8": 0.09565217391304345,
            "high_confidence_safe_coverage@0.8": 0.08167613636363638,
            "positive_rate": 0.3380681818181819,
            "mean_probability": 0.2151535858899647,
            "count": 1408,
            "scene_count": 142
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.06514706982923234,
              0.11439728678219573
            ],
            "nll_95": [
              0.1972825471798324,
              0.42238148954313987
            ],
            "ece_95": [
              0.050321817602256166,
              0.11366648715248898
            ],
            "false_safe_rate@0.8_95": [
              0.009958684844219157,
              0.4139420620549496
            ]
          }
        }
      ],
      "by_source": [
        {
          "scope": "source",
          "source_dataset": "3RScan",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.05909080808771263,
            "nll": 0.1894277904546879,
            "ece": 0.06675810888196806,
            "false_safe_rate@0.8": 0.11211680625806318,
            "high_confidence_safe_coverage@0.8": 0.28392929821501256,
            "positive_rate": 0.3444793301936159,
            "mean_probability": 0.35200108329260443,
            "count": 417,
            "scene_count": 21
          },
          "query_weighted": {
            "brier": 0.041275598655119386,
            "nll": 0.14853048882118092,
            "ece": 0.037215171673265206,
            "false_safe_rate@0.8": 0.030150753768844223,
            "high_confidence_safe_coverage@0.8": 0.4772182254196642,
            "positive_rate": 0.5467625899280575,
            "mean_probability": 0.5351919147333155,
            "count": 417,
            "scene_count": 21
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.03686197133235068,
              0.0846125485529467
            ],
            "nll_95": [
              0.13383013669938806,
              0.25545481787979046
            ],
            "ece_95": [
              0.05085476235982962,
              0.11055572528404108
            ],
            "false_safe_rate@0.8_95": [
              0.013567243316107673,
              0.2674699649155115
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ARKitScenes",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.0995420401166204,
            "nll": 0.297028994597547,
            "ece": 0.07875599867012181,
            "false_safe_rate@0.8": 0.11782996780801874,
            "high_confidence_safe_coverage@0.8": 0.2086080586080586,
            "positive_rate": 0.2864138176638177,
            "mean_probability": 0.3206866292094223,
            "count": 381,
            "scene_count": 26
          },
          "query_weighted": {
            "brier": 0.09320920924001022,
            "nll": 0.29619623369033965,
            "ece": 0.059800337327324475,
            "false_safe_rate@0.8": 0.1290322580645161,
            "high_confidence_safe_coverage@0.8": 0.3254593175853019,
            "positive_rate": 0.3674540682414698,
            "mean_probability": 0.4272544055687944,
            "count": 381,
            "scene_count": 26
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.07037593132002412,
              0.1294032672294351
            ],
            "nll_95": [
              0.2210755254882143,
              0.37583099936630315
            ],
            "ece_95": [
              0.049437318547200536,
              0.13437052679879324
            ],
            "false_safe_rate@0.8_95": [
              0.04572299402444521,
              0.19857995578836737
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "Matterport3D",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.11325197695732266,
            "nll": 0.373170666198726,
            "ece": 0.07727104980796073,
            "false_safe_rate@0.8": 0.1270761229857926,
            "high_confidence_safe_coverage@0.8": 0.4819799666786085,
            "positive_rate": 0.5546465160005059,
            "mean_probability": 0.5463966202596048,
            "count": 1104,
            "scene_count": 31
          },
          "query_weighted": {
            "brier": 0.10006266771923142,
            "nll": 0.32806546372808265,
            "ece": 0.06371828744833175,
            "false_safe_rate@0.8": 0.08858603066439523,
            "high_confidence_safe_coverage@0.8": 0.5317028985507246,
            "positive_rate": 0.6286231884057971,
            "mean_probability": 0.604877256897669,
            "count": 1104,
            "scene_count": 31
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.08426643293444447,
              0.14777310909422167
            ],
            "nll_95": [
              0.2690401111565067,
              0.5086136623227996
            ],
            "ece_95": [
              0.0466681713480608,
              0.12117382895557843
            ],
            "false_safe_rate@0.8_95": [
              0.05987185761948051,
              0.21043215051010203
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ScanNet",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.09452215420457684,
            "nll": 0.2909505545548017,
            "ece": 0.047637761858409344,
            "false_safe_rate@0.8": 0.09624556462273838,
            "high_confidence_safe_coverage@0.8": 0.3303003422428248,
            "positive_rate": 0.4155103866041366,
            "mean_probability": 0.4277804284544696,
            "count": 1008,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.09722725891138204,
            "nll": 0.30478728326317595,
            "ece": 0.04644179549132823,
            "false_safe_rate@0.8": 0.07859078590785908,
            "high_confidence_safe_coverage@0.8": 0.36607142857142855,
            "positive_rate": 0.45932539682539675,
            "mean_probability": 0.4626765209376737,
            "count": 1008,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.07297489310148061,
              0.11811749774997941
            ],
            "nll_95": [
              0.2294424151483238,
              0.3602558140224409
            ],
            "ece_95": [
              0.032750235642768515,
              0.09274764050924086
            ],
            "false_safe_rate@0.8_95": [
              0.0390519624745518,
              0.1630550882437876
            ]
          }
        },
        {
          "scope": "source",
          "source_dataset": "ZInD",
          "radius_cells": null,
          "scene_weighted": {
            "brier": 0.08074347971804055,
            "nll": 0.3037141185061457,
            "ece": 0.0925961034743979,
            "false_safe_rate@0.8": 0.012596850389253946,
            "high_confidence_safe_coverage@0.8": 0.5071049538965077,
            "positive_rate": 0.6592677331841419,
            "mean_probability": 0.566671629709744,
            "count": 1314,
            "scene_count": 32
          },
          "query_weighted": {
            "brier": 0.08868926670631325,
            "nll": 0.3357728430691239,
            "ece": 0.10643366006424378,
            "false_safe_rate@0.8": 0.011204481792717085,
            "high_confidence_safe_coverage@0.8": 0.54337899543379,
            "positive_rate": 0.7161339421613394,
            "mean_probability": 0.6097002820970956,
            "count": 1314,
            "scene_count": 32
          },
          "scene_bootstrap_95": {
            "unit": "(source_dataset, scene_id)",
            "samples": 2000,
            "brier_95": [
              0.05373154659611149,
              0.12029746782991668
            ],
            "nll_95": [
              0.17994301439572824,
              0.5214334896901949
            ],
            "ece_95": [
              0.06351574060856315,
              0.13878155038153955
            ],
            "false_safe_rate@0.8_95": [
              0.0043637559809863715,
              0.021265613239081965
            ]
          }
        }
      ],
      "radius_monotonicity": {
        "query_count": 1408,
        "violating_queries": 0,
        "violation_rate": 0.0,
        "maximum_probability_increase": 0.0,
        "rule": "predicted reachability must be non-increasing as footprint radius grows"
      }
    }
  ],
  "claim_boundary": "Validation-only baseline diagnostics on the bounded provenance split. The test split is still locked, and these numbers are not final paper results."
};
