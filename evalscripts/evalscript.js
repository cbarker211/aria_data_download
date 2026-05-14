//VERSION=3

function setup() {
    return {
        input: [{
            datasource: "S2L1C",
            bands: ["sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean", "dataMask", "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12"],
            units: "DN"
        }],
        output: [{
            bands: 16,
            sampleType: "FLOAT32"
        }],
    };
}

function evaluatePixel(sample) {
    return [
        sample.sunAzimuthAngles,
        sample.sunZenithAngles,
        sample.viewAzimuthMean,
        sample.viewZenithMean,
        sample.dataMask,
        sample.B01,
        sample.B02,
        sample.B03,
        sample.B04,
        sample.B05,
        sample.B06,
        sample.B07,
        sample.B08,
        sample.B09,
        sample.B11,
        sample.B12,
    ];
}