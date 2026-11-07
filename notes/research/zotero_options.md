# Zotero filing options: "CS2 CSDS Round Win Probability"

Research only, checked 2026-10-12. Nothing was installed. No API key or credential was read, stored, requested or printed.

## Machine state

- Zotero desktop is installed at `C:\Program Files\Zotero\zotero.exe` and is running (3 processes named `zotero`).
- Connector ping `http://127.0.0.1:23119/connector/ping` failed with "The underlying connection was closed: The connection was closed unexpectedly." The connector did not answer.

## Official docs: what could and could not be confirmed

- **Local connector** (https://www.zotero.org/support/dev/client_coding/connector_http_server): lists endpoints such as `/connector/ping` and `/connector/saveItems` but does not describe them. It does not say whether the connector can create collections or save into an existing collection. **Not confirmed.**
- **Web API Basics** (https://www.zotero.org/support/dev/web_api/v3/basics): the userID is different from the username and is shown on the zotero.org API Keys page. Non-public libraries require an API key. Local API writes (Zotero 10+) require a local key granted by the user at runtime, which is unrelated to a zotero.org key; without it writes return 401.
- **Web API OAuth page** (https://www.zotero.org/support/dev/web_api/v3/oauth): API keys are created in zotero.org account settings, and a write-access option is available when creating one.
- **Web API Write Requests** (https://www.zotero.org/support/dev/web_api/v3/write_requests): a collection is created by POST to `<userOrGroupPrefix>/collections` (up to 50 per request). An item is filed into a collection by listing the collection key in the item's `collections` array. Omitting a key removes the item from that collection.
- **Not found (HTTP 404)**: https://www.zotero.org/support/dev/client_api, https://www.zotero.org/support/dev/zotero_connector, https://www.zotero.org/support/dev/client_coding/local_api, https://www.zotero.org/support/dev/local_api. Not read.

## Options, ranked by how little the user must do

1. **cookjohn/zotero-mcp** (Zotero plugin with built-in MCP server, 1,226 stars, TypeScript)
   - User must do: install the plugin in Zotero, restart Zotero, enable the server in preferences. Create the empty collection once in the Zotero UI.
   - Credentials: none. Local only.
   - Can: `add_by_identifier` accepts a `collectionKey`, so newly imported items (by DOI, URL, etc.) can be filed into an existing collection.
   - Cannot: create collections, or move existing items into a collection (per the repo's tool list).
   - Source: https://github.com/cookjohn/zotero-mcp

2. **54yyyu/zotero-mcp** (Python MCP server, 5,344 stars)
   - User must do: enable "Allow other applications on this computer to communicate with Zotero" (Settings, Advanced). On Zotero 10+, run `zotero-mcp authorize-local` once and choose "Always Allow".
   - Credentials: none for local mode. Web mode instead needs an API key and library ID.
   - Can: writes through the running Zotero 10+ client. The README says "manage collections", but creating a collection and filing items were not explicitly confirmed.
   - Source: https://github.com/54yyyu/zotero-mcp

3. **Zotero Web API called directly** (official; no MCP server needed)
   - User must provide: (a) the numeric userID from the zotero.org API Keys page, and (b) an API key created with write access. These are credentials. They must be supplied by the user; they were not requested, stored or printed here.
   - Can: create the collection (POST `/users/<userID>/collections`) and file items (POST `/users/<userID>/items` with the collection key in `collections`). Confirmed in the official docs.
   - Sources: https://www.zotero.org/support/dev/web_api/v3/write_requests, https://www.zotero.org/support/dev/web_api/v3/basics, https://www.zotero.org/support/dev/web_api/v3/oauth

Option 3 is the only path confirmed to create the collection programmatically, but it needs credentials from the user.

## Top 5 Zotero MCP servers or plugins by stars (GitHub repo search "zotero mcp", 2026-10-12)

| Repo | Stars | Needs on this machine | Writes? |
|---|---|---|---|
| 54yyyu/zotero-mcp | 5,344 | Desktop running plus local authorization (or web key and library ID) | Yes; collection creation not confirmed |
| papersgpt-for-zotero | 2,667 | Zotero desktop plugin; LLM key only for AI features | No collection creation or filing described |
| cookjohn/zotero-mcp | 1,226 | Zotero desktop plugin with server enabled; no key | Adds items into an existing collection only |
| kaliaboi/mcp-zotero | 164 | Zotero Cloud API key and userID | Read-only |
| kujenga/zotero-mcp | 162 | Desktop local API or web key | Read-only |

Other results, not ranked: gyger/mcp-pyzotero (56 stars, desktop local API, read-only); yilewang/llm-for-zotero (3,278 stars, research agent plugin, not evaluated); PDFMathTranslate and claude-scholar (not Zotero MCP servers).

## Meridian zotero_search

Read-only. Its schema has only `query`, `library_id`, `library_type` (user or group), `limit` and `sort_by`. It has no write parameters and cannot file items or create collections.

## Open items

- Why the connector ping fails (port 23119 is open but drops the connection). Not diagnosed.
- Whether the connector can create collections or save into an existing one. The official connector page does not say.
- Whether 54yyyu/zotero-mcp can create a collection. Not confirmed.
