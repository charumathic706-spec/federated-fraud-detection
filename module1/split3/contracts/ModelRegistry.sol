// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

// =============================================================================
// contracts/ModelRegistry.sol
// Smart contract for Federated Learning Blockchain Governance
//
// Deployed on Ganache local Ethereum blockchain.
// Called from Python via web3.py after every FL round.
//
// What this contract enforces ON-CHAIN:
//   1. Each round's model hash is stored permanently
//   2. Hash chain is VALIDATED on registration - if prevBlockHash does not
//      match the stored blockHash of round-1, the transaction REVERTS.
//      This means tampered rounds CANNOT be registered.
//   3. Tamper alerts are permanent - cannot be deleted
//   4. Client quarantine decisions are on-chain
//   5. Full audit trail - GDPR Article 30
// =============================================================================

contract ModelRegistry {

    address public owner;

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "Only FL server can register");
        _;
    }

    // ── Structs ───────────────────────────────────────────────────────────────

    struct ModelRecord {
        uint256 round;
        string  modelHash;
        string  blockHash;
        string  prevBlockHash;
        uint256 globalF1;    // F1 * 1000000 (e.g. 0.836 stored as 836000)
        uint256 globalAUC;
        uint256 timestamp;
        bool    exists;
    }

    struct TamperAlert {
        uint256 id;
        uint256 round;
        string  alertType;
        string  detail;
        string  severity;
        uint256 timestamp;
    }

    struct AuditEvent {
        uint256 id;
        string  eventType;
        uint256 round;
        string  actor;
        string  dataJson;
        uint256 timestamp;
    }

    // ── Storage ───────────────────────────────────────────────────────────────

    mapping(uint256 => ModelRecord)  private models;
    mapping(uint256 => TamperAlert)  private alerts;
    mapping(uint256 => AuditEvent)   private auditLog;
    mapping(int256  => bool)         public  quarantined;

    uint256 public roundCount;
    uint256 public alertCount;
    uint256 public auditCount;

    // ── Events ────────────────────────────────────────────────────────────────

    event ModelRegistered(
        uint256 indexed round,
        string  modelHash,
        string  blockHash,
        uint256 f1,
        uint256 ts
    );
    event TamperAlertRaised(
        uint256 indexed id,
        uint256 indexed round,
        string  alertType,
        string  severity,
        uint256 ts
    );
    event AuditEventLogged(
        uint256 indexed id,
        string  eventType,
        uint256 indexed round,
        uint256 ts
    );
    event ClientQuarantined(
        int256  indexed clientId,
        uint256 round,
        uint256 consecutiveFlags,
        uint256 ts
    );

    // =========================================================================
    // WRITE FUNCTIONS
    // =========================================================================

    // ── registerModel ─────────────────────────────────────────────────────────
    // Core governance function. Called once per FL round.
    //
    // CHAIN VALIDATION (the key part):
    //   For round > 1: checks that prevBlockHash == models[round-1].blockHash
    //   If this fails, the ENTIRE TRANSACTION IS REVERTED.
    //   This means: you cannot register round 5 unless rounds 1-4 are valid.
    //   Any tampering with a historical round breaks all future registrations.
    function registerModel(
        uint256         round,
        string  memory  modelHash,
        string  memory  blockHash,
        string  memory  prevBlockHash,
        uint256         globalF1,
        uint256         globalAUC,
        int256[] memory trustedClients,
        int256[] memory flaggedClients
    ) external onlyOwner {
        require(round > 0,             "Round must be > 0");
        require(!models[round].exists, "Round already registered");

        // HASH CHAIN VALIDATION - this is what makes it a real blockchain
        if (round > 1) {
            require(
                models[round - 1].exists,
                "Previous round not found - must register in order"
            );
            require(
                keccak256(bytes(prevBlockHash)) ==
                keccak256(bytes(models[round - 1].blockHash)),
                "Hash chain broken: prevBlockHash mismatch"
            );
        }

        models[round] = ModelRecord({
            round:         round,
            modelHash:     modelHash,
            blockHash:     blockHash,
            prevBlockHash: prevBlockHash,
            globalF1:      globalF1,
            globalAUC:     globalAUC,
            timestamp:     block.timestamp,
            exists:        true
        });

        roundCount++;

        emit ModelRegistered(round, modelHash, blockHash, globalF1, block.timestamp);
    }

    // ── raiseTamperAlert ──────────────────────────────────────────────────────
    function raiseTamperAlert(
        uint256        round,
        string  memory alertType,
        string  memory detail,
        string  memory severity
    ) external onlyOwner returns (uint256) {
        alertCount++;
        alerts[alertCount] = TamperAlert({
            id:        alertCount,
            round:     round,
            alertType: alertType,
            detail:    detail,
            severity:  severity,
            timestamp: block.timestamp
        });
        emit TamperAlertRaised(alertCount, round, alertType, severity, block.timestamp);
        return alertCount;
    }

    // ── appendAuditEvent ─────────────────────────────────────────────────────
    function appendAuditEvent(
        string  memory eventType,
        uint256        round,
        string  memory actor,
        string  memory dataJson
    ) external onlyOwner returns (uint256) {
        auditCount++;
        auditLog[auditCount] = AuditEvent({
            id:        auditCount,
            eventType: eventType,
            round:     round,
            actor:     actor,
            dataJson:  dataJson,
            timestamp: block.timestamp
        });
        emit AuditEventLogged(auditCount, eventType, round, block.timestamp);
        return auditCount;
    }

    // ── quarantineClient ──────────────────────────────────────────────────────
    function quarantineClient(
        int256         clientId,
        uint256        round,
        uint256        consecutiveFlags,
        string  memory reason
    ) external onlyOwner {
        quarantined[clientId] = true;
        emit ClientQuarantined(clientId, round, consecutiveFlags, block.timestamp);
    }

    // =========================================================================
    // READ FUNCTIONS (view - no gas, no transaction)
    // =========================================================================

    function getModel(uint256 round)
        external view
        returns (
            string memory modelHash,
            string memory blockHash,
            string memory prevBlockHash,
            uint256       globalF1,
            uint256       globalAUC,
            uint256       timestamp,
            bool          exists
        )
    {
        ModelRecord storage r = models[round];
        return (r.modelHash, r.blockHash, r.prevBlockHash,
                r.globalF1, r.globalAUC, r.timestamp, r.exists);
    }

    // ── verifyModelHash ───────────────────────────────────────────────────────
    // Called to prove a model is genuine - returns true if hash matches ledger
    function verifyModelHash(uint256 round, string memory claimedHash)
        external view
        returns (bool isValid, string memory storedHash)
    {
        require(models[round].exists, "Round not found");
        storedHash = models[round].modelHash;
        isValid    = keccak256(bytes(storedHash)) == keccak256(bytes(claimedHash));
    }

    // ── verifyFullChain ───────────────────────────────────────────────────────
    // Verifies the entire hash chain from round 1 to roundCount
    // Returns (intact, brokenAtRound) - brokenAtRound=0 means no break
    function verifyFullChain()
        external view
        returns (bool intact, uint256 brokenAtRound)
    {
        intact = true;
        brokenAtRound = 0;
        for (uint256 r = 2; r <= roundCount; r++) {
            if (
                !models[r].exists ||
                !models[r-1].exists ||
                keccak256(bytes(models[r].prevBlockHash)) !=
                keccak256(bytes(models[r-1].blockHash))
            ) {
                return (false, r);
            }
        }
    }

    function getAlert(uint256 id)
        external view
        returns (uint256 round, string memory alertType,
                 string memory severity, uint256 timestamp)
    {
        TamperAlert storage a = alerts[id];
        return (a.round, a.alertType, a.severity, a.timestamp);
    }

    function getAuditEvent(uint256 id)
        external view
        returns (string memory eventType, uint256 round,
                 string memory actor, string memory dataJson, uint256 timestamp)
    {
        AuditEvent storage e = auditLog[id];
        return (e.eventType, e.round, e.actor, e.dataJson, e.timestamp);
    }

    function getRoundCount() external view returns (uint256) { return roundCount; }
    function getAlertCount() external view returns (uint256) { return alertCount; }
    function getAuditCount() external view returns (uint256) { return auditCount; }
}
