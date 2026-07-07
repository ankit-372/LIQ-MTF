import logging
from src.shared.event_bus import EventBus
from src.scenario import ScenarioManager
from src.pattern_matcher import PatternMatcher
from src.risk_rules import RiskEngine
from src.journal import JournalManager
from src.portfolio import PortfolioTracker
from src.circuit_breaker import CircuitBreaker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

class DecisionAgent:
    """
    DecisionAgent acts as the master coordinator for the decision pipeline.
    It links ScenarioManager, PatternMatcher, RiskEngine, and JournalManager together
    via the central EventBus.
    """
    def __init__(
        self, 
        portfolio: PortfolioTracker, 
        circuit_breaker: CircuitBreaker, 
        db_journal: str = "src/data/journal.db", 
        db_matcher: str = "models/signals_history.db", 
        base_size: float = 1000.0
    ):
        self.portfolio = portfolio
        self.circuit_breaker = circuit_breaker
        
        self.scenario_manager = ScenarioManager()
        self.pattern_matcher = PatternMatcher(db_path=db_matcher)
        self.risk_engine = RiskEngine(base_size=base_size)
        self.journal = JournalManager(db_path=db_journal)
        
        # Subscribe to pipeline events
        EventBus.subscribe("SCENARIO_CREATED", self.on_scenario_created)
        EventBus.subscribe("AGENT_DECISION_MADE", self.on_decision_made)
        EventBus.subscribe("TRADE_CLOSED", self.on_trade_closed)
        
        logging.info("DecisionAgent: Subscribed to 'SCENARIO_CREATED', 'AGENT_DECISION_MADE', and 'TRADE_CLOSED'")

    def on_scenario_created(self, data: dict):
        """
        Triggered when a new market scenario snapshot is created.
        Queries the pattern matcher, evaluates risk rules, and publishes decision.
        """
        scenario = data["scenario"]
        logging.info(f"DecisionAgent: Packaging scenario for timestamp {scenario.get('timestamp')}")
        
        # 1. Match patterns
        pattern_matches = self.pattern_matcher.match_scenario(scenario)
        logging.info(f"DecisionAgent: Matches found: {pattern_matches.get('matches')} (Expectancy: {pattern_matches.get('expectancy', 0.0):.4f})")
        
        # 2. Evaluate risk rules
        self.risk_engine.evaluate_rules(
            scenario=scenario,
            portfolio=self.portfolio,
            pattern_matches=pattern_matches,
            circuit_breaker=self.circuit_breaker,
            journal=self.journal
        )

    def on_decision_made(self, data: dict):
        """
        Triggered when a decision has been evaluated by the RiskEngine.
        Logs the decision to the journal.
        """
        logging.info(f"DecisionAgent: Decision made. Override: {data.get('override_triggered')}, Final Size: ${data.get('final_size'):.2f}")
        self.journal.log_decision(data)

    def on_trade_closed(self, data: dict):
        """
        Triggered when a trade is closed.
        Updates the corresponding journal entry with the realized trade outcome.
        """
        trade_id = data.get("trade_id")
        pnl = float(data.get("realized_pnl", data.get("exit_pnl", 0.0)))
        exit_reason = data.get("exit_reason", "STOP_LOSS")
        
        logging.info(f"DecisionAgent: Trade closed: {trade_id}. PnL: ${pnl:.2f}. Reason: {exit_reason}")
        self.journal.update_outcome(trade_id, pnl, exit_reason)
