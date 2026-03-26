//VERSION=3

function setup() {
    return {
        input: [{
            datasource: "S2L1C",
            bands: ["sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean", "dataMask", "B01", "B02", "B03", "B04", "B08"],
            units: "DN"
        }],
        output: [{
            bands: 10,
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
        sample.B08
    ];
}