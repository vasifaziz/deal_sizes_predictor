const form = document.getElementById("prediction-form");
const resultBox = document.getElementById("result-box");
const predictionElement = document.getElementById("prediction");
const confidenceElement = document.getElementById("confidence");
const tableBody = document.getElementById("input-table-body");

// The frontend is commonly opened from file:// or a separate static server.
// In both cases, a relative URL would incorrectly target the frontend origin.
const API_BASE_URL =
    window.DEAL_PREDICTOR_API_URL ||
    (window.location.protocol === "http:" &&
    window.location.port === "8000"
        ? window.location.origin
        : "dealsizespredictor-production.up.railway.app");

const rangeInputs = [
    ["quantity", "quantity-value"],
    ["days", "days-value"],
    ["msrp", "msrp-value"],
    ["line-number", "line-number-value"]
];

rangeInputs.forEach(([inputId, valueId]) => {
    const input = document.getElementById(inputId);
    const output = document.getElementById(valueId);

    input.addEventListener("input", () => {
        output.textContent = input.value;
    });
});

document.getElementById("order-date").value =
    new Date().toISOString().split("T")[0];

function getFormData() {
    return {
        QUANTITYORDERED: Number(
            document.getElementById("quantity").value
        ),

        PRICEEACH: Number(
            document.getElementById("price").value
        ),

        DAYS_SINCE_LASTORDER: Number(
            document.getElementById("days").value
        ),

        MSRP: Number(
            document.getElementById("msrp").value
        ),

        ORDERLINENUMBER: Number(
            document.getElementById("line-number").value
        ),

        PRODUCTLINE:
            document.getElementById("product-line").value,

        COUNTRY:
            document.getElementById("country").value,

        ORDERDATE:
            document.getElementById("order-date").value
    };
}

function displayInputData(data) {
    tableBody.innerHTML = "";

    Object.entries(data).forEach(([key, value]) => {
        const row = document.createElement("tr");

        row.innerHTML = `
            <td>${key}</td>
            <td>${value}</td>
        `;

        tableBody.appendChild(row);
    });
}

function displayPrediction(prediction, confidence) {
    resultBox.classList.remove("hidden");

    predictionElement.className = "";

    if (prediction === "Small") {
        predictionElement.textContent =
            "The Deal Size is Small";

        predictionElement.classList.add("small");

    } else if (prediction === "Medium") {
        predictionElement.textContent =
            "The Deal Size is Medium";

        predictionElement.classList.add("medium");

    } else {
        predictionElement.textContent =
            "The Deal Size is Large";

        predictionElement.classList.add("large");
    }

    confidenceElement.textContent =
        `Probability of Deal Size is ${prediction}: ` +
        `${(confidence * 100).toFixed(2)}%`;
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const inputData = getFormData();

    displayInputData(inputData);

    resultBox.classList.remove("hidden");
    predictionElement.textContent = "Predicting...";
    confidenceElement.textContent = "";

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(inputData)
        });

        if (!response.ok) {
            let message = "Prediction request failed.";
            try {
                const errorResult = await response.json();
                message = errorResult.detail || message;
            } catch {
                // Keep the generic message when the server does not return JSON.
            }
            throw new Error(message);
        }

        const result = await response.json();

        displayPrediction(
            result.predicted_dealsize,
            result.confidence
        );

    } catch (error) {
        predictionElement.textContent =
            "Unable to get prediction.";

        confidenceElement.textContent =
            error.message;
    }
});
