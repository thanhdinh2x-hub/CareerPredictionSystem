import express from "express";
import axios from "axios";

console.log("Loading server.ts");

const app = express();

app.use(
    express.static("public")
);

app.set(
    "view engine",
    "pug"
);


app.set(
    "views",
    "./views"
);


app.use(
    express.urlencoded({
        extended: true
    })
);



app.get(
    "/",
    (req, res) => {

        res.render(
            "index"
        );

    }
);


app.post(
    "/predict",
    async (req, res) => {

        try {

            const response = await axios.post(
                "http://127.0.0.1:8000/predict",
                {
                    title: req.body.title,
                    location: req.body.location,
                    description: req.body.description,
                    function: req.body.function,
                    industry: req.body.industry
                }
            );


            res.render(
                "index",
                {
                    result: response.data.career_level
                }
            );


        } catch(error) {

            console.log(error);

            res.render(
                "index",
                {
                    result: "Prediction failed"
                }
            );

        }

    }
);

app.listen(
    3000,
    () => {

        console.log(
            "Server running at http://localhost:3000"
        );

    }
);