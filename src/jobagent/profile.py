"""The candidate profile: the structured, user-reviewed version of the resume (config/profile.json).

This one schema is used for Claude's resume extraction, for storage, and for rendering.
Bullet IDs are derived from position (`exp1.2` = 1st role, 2nd bullet) rather than stored,
so the user can edit profile.json freely; tailoring cites these IDs to prove it invented nothing.
"""

import json
from pathlib import Path

from pydantic import BaseModel


class Link(BaseModel):
    label: str
    url: str


class Role(BaseModel):
    company: str
    title: str
    location: str
    start: str
    end: str
    bullets: list[str]


class Project(BaseModel):
    name: str
    link: str
    bullets: list[str]


class Education(BaseModel):
    institution: str
    degree: str
    start: str
    end: str
    score: str


class SkillGroup(BaseModel):
    name: str
    skills: list[str]


class Profile(BaseModel):
    name: str
    email: str
    phone: str
    location: str
    links: list[Link]
    headline: str
    summary: str
    skills: list[SkillGroup]
    experience: list[Role]
    projects: list[Project]
    education: list[Education]
    certifications: list[str]
    achievements: list[str]

    def bullets(self) -> dict[str, str]:
        """Every experience/project bullet keyed by its stable ID."""
        ids: dict[str, str] = {}
        for i, role in enumerate(self.experience, 1):
            for j, text in enumerate(role.bullets, 1):
                ids[f"exp{i}.{j}"] = text
        for i, project in enumerate(self.projects, 1):
            for j, text in enumerate(project.bullets, 1):
                ids[f"proj{i}.{j}"] = text
        return ids

    def all_skills(self) -> list[str]:
        return [skill for group in self.skills for skill in group.skills]

    def is_placeholder(self) -> bool:
        """True for the bundled sample profile (example.com email)."""
        return self.email.endswith("@example.com")


def load_profile(path: Path) -> Profile:
    return Profile.model_validate_json(path.read_text())


def save_profile(profile: Profile, path: Path) -> None:
    path.write_text(json.dumps(profile.model_dump(), indent=2, ensure_ascii=False) + "\n")
