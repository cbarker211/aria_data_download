//VERSION=3

function setup() {
    return {
        input: [{
            datasource: "S2L1C",
            bands: ["sunZenithAngles", "viewZenithMean", "dataMask", "BAND_PLACEHOLDER", "B04"],
            units: "DN"
        }],
        output: [{
            bands: 5,
            sampleType: "FLOAT32"
        }],
    };
}

function evaluatePixel(sample) {
    return [sample.sunZenithAngles, sample.viewZenithMean, sample.dataMask, sample.BAND_PLACEHOLDER, sample.B04];
}