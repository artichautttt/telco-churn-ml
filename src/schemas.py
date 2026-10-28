"""
schemas.py -- schemas Pydantic de l'API de prediction.

`CustomerFeatures` reprend exactement les colonnes du dataset d'entrainement
(hors customerID et Churn), avec les memes noms, pour que le DataFrame
construit a partir d'une requete passe tel quel dans le pipeline sauvegarde.

Validation stricte pour eviter les erreurs silencieuses : le OneHotEncoder
du pipeline utilise handle_unknown="ignore", donc une categorie mal
orthographiee ("month-to-month", "Fiber") serait encodee en zeros sans
erreur -- d'ou les Literal, qui la rejettent en 422 avant le modele.
"""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetOption = Literal["Yes", "No", "No internet service"]

INTERNET_DEPENDENT = (
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
)


class CustomerFeatures(BaseModel):
    # strict : pas de conversion implicite ("12" -> 12, true -> 1) ;
    # extra="forbid" : un champ inconnu ou mal nomme est une erreur.
    model_config = ConfigDict(strict=True, extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Annotated[int, Field(ge=0, le=1)]
    Partner: YesNo
    Dependents: YesNo
    tenure: Annotated[int, Field(ge=0, le=120, description="Anciennete en mois")]
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetOption
    OnlineBackup: InternetOption
    DeviceProtection: InternetOption
    TechSupport: InternetOption
    StreamingTV: InternetOption
    StreamingMovies: InternetOption
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: Annotated[float, Field(gt=0, le=500)]
    TotalCharges: Annotated[float, Field(ge=0, le=50_000)]

    @model_validator(mode="after")
    def check_service_consistency(self):
        """Coherence entre services, telle qu'observee dans le dataset."""
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError(
                "MultipleLines doit valoir 'No phone service' si et seulement si PhoneService == 'No'"
            )
        no_internet = self.InternetService == "No"
        for field in INTERNET_DEPENDENT:
            if no_internet != (getattr(self, field) == "No internet service"):
                raise ValueError(
                    f"{field} doit valoir 'No internet service' si et seulement si "
                    f"InternetService == 'No'"
                )
        return self


class PredictionResponse(BaseModel):
    churn_probability: float = Field(ge=0, le=1, description="Probabilite de churn calibree")
    churn_prediction: Literal[0, 1] = Field(description="1 si churn_probability >= threshold")
    threshold: float = Field(gt=0, lt=1, description="Seuil de decision (matrice de couts)")
