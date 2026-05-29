"""Pydantic-схема заявок на курсы ДПО."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

CITIES: set[str] = {
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Ростов-на-Дону",
    "Воронеж",
}

SPECIALITIES = [
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "юрист",
    "менеджер",
    "IT-специалист",
    "психолог",
    "экономист",
    "социальный работник",
]

DESIRED_COURSES = [
    "цифровая педагогика",
    "медицинская реабилитация",
    "управление проектами",
    "налогообложение",
    "информационная безопасность",
    "корпоративное право",
    "data science для бизнеса",
    "управление персоналом",
]

Speciality = Literal[
    "учитель",
    "врач",
    "инженер",
    "бухгалтер",
    "юрист",
    "менеджер",
    "IT-специалист",
    "психолог",
    "экономист",
    "социальный работник",
]

DesiredCourse = Literal[
    "цифровая педагогика",
    "медицинская реабилитация",
    "управление проектами",
    "налогообложение",
    "информационная безопасность",
    "корпоративное право",
    "data science для бизнеса",
    "управление персоналом",
]


class Address(BaseModel):
    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


class Application(BaseModel):
    full_name: str
    age: int = Field(ge=22, le=65)
    address: Address
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    @field_validator("graduation_year")
    @classmethod
    def graduation_matches_age(cls, v: int, info) -> int:
        age = info.data.get("age")
        if age is not None:
            current_year = date.today().year
            if v > current_year:
                raise ValueError(
                    f"Год окончания {v} позже текущего ({current_year})"
                )
            if v > current_year - age + 22:
                raise ValueError(
                    f"Год окончания {v} несовместим с возрастом {age}: "
                    f"слишком позднее окончание для данного возраста"
                )
        return v

    @property
    def city(self) -> str:
        return self.address.city
