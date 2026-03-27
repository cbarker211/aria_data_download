//VERSION=3

function setup() {
    return {
        input: [{
            datasource: "S2L1C",
            bands: ["B01"],
            units: "DN"
        }],
        output: [{
            bands: 1,
            sampleType: "FLOAT32"
        }],
    };
}

function evaluatePixel(sample) {
    return [
        sample.B01
    ];
}