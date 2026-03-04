//VERSION=3

function setup() {
    return {
        input: [{
            datasource: "S2L2A",
            bands: ["SCL"],
            units: "DN"
        }],
        output: [{
            bands: 1,
            sampleType: "UINT8"
        }],
    };
}

function evaluatePixel(sample) {
    return [sample.SCL];
}