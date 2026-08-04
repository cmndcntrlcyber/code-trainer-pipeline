from .base import AbliterationTechnique
from .obliteratus_tech import ObliteratusTechnique
from .nousresearch_tech import NousResearchTechnique
from .abliterix_tech import AbliterixTechnique

TECHNIQUES = {
    "obliteratus": ObliteratusTechnique,
    "nousresearch": NousResearchTechnique,
    "abliterix": AbliterixTechnique,
}
