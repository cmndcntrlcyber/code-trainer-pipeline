from .config import FARCAConfig
from .claim_extractor import ClaimExtractor, AtomicClaim, ExtractionResult
from .claim_verifier import ClaimVerifier, VerificationResult
from .counterfactual_attribution import CounterfactualAttributor, ReliabilityResult
from .reward import compute_farca_reward, FARCAReward
from .advantage_reshaper import reshape_advantages_batch
from .pipeline import FARCAPipeline
