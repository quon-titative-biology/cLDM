This directory contains the example `.tiff` and `.json` under the `/example_data_1/` folder necessary to run the example process
- Root folders containg data should be formatted as `batch_VAST_HSDs_pooled_humanized_age_genotype_plate`

Naming Convention Requirements:
- Directory needs to contain the plate number as the final character of the name
- Image `.tiff` naming convention: `batch_age_(tail or head)_W_ID_plate_Angle_rot.tiff`
- JSON `.json` naming convention: `batch_age_(tail or head)_W_ID_plate_Angle_rot__SHAPES.json`

JSON File Requirements:
The preprocessing scripts are build around the FishInspectro JSON files as such all JSON files should be in the same format with the first 3 features as such:
- `contourDV_net`, `yolkDV_net`, `eye1DV_net`


Example directory stucture:
```
├── 2022.11.11_VAST_HSDs_pooled_humanized_5dpf_arhgap11_plate1/
│   ├── 2022.11.11_5dpf_tail_W_G01_1_4_rot__SHAPES.json
│   ├── 2022.11.11_5dpf_tail_W_G01_1_4_rot.tiff
```

Example JSON file structure
```
{
	"version": 0.99,
	"enabled": 1,
	"imageDimensions": [1024,200,3],
	"contourDV_net": {
		"mode": "auto",
		"shape": {
			"name": "fineContour",
			"x":...
			"y":...
		},
		"regionprops": {
       ...
		}
	},
	"eye1DV_net": {
		"shape": {
			"name": "fineContour",
			"x": ...
			"y": ...
		},
		"regionprops": {
       ...
		}
	},
	"yolkDV_net": {
		"shape": {
			"name": "fineContour",
			"x": ...
			"y": ...
		},
		"regionprops": {
       ...
		}
	}
}
```

