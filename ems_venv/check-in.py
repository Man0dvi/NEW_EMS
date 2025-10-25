@mcp.tool
def comparative_analysis_tool(
    knowledge_base: List[Dict[str, Any]], 
    query_topic: str
) -> List[Dict[str, Any]]:
    """
    Groups and compares semantically similar text chunks to find consensus, 
    reinforcement, or contradiction on a specific topic.

    Args:
        knowledge_base: The list of enriched document chunks from the Retrieval Agent (Agent 2).
        query_topic: The specific analytical question driving the comparison (e.g., 'NMCP funding sources').

    Returns:
        A list of grouped findings, where each finding contains a summary of consensus 
        and the supporting chunk IDs and sources.
    """
    if len(knowledge_base) < 2:
        return [{"status": "Skipped", "message": "Requires at least 2 chunks for comparison."}]
        
    df = pd.DataFrame(knowledge_base)
    texts = df['text_content'].tolist()
    
    # 1. Generate Embeddings and Similarity Matrix
    text_embeddings = EMBEDDING_MODEL.encode(texts, convert_to_tensor=False)
    similarity_matrix = cosine_similarity(text_embeddings)
    
    # 2. Group similar chunks (Semantic Agreement)
    comparison_groups = {}
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i, j] >= SIMILARITY_THRESHOLD:
                # Group chunks by their IDs
                group_key = frozenset(sorted([df.loc[i, 'chunk_id'], df.loc[j, 'chunk_id']]))
                
                if group_key not in comparison_groups:
                    comparison_groups[group_key] = []
                
                # Add unique items to the group (simplified insertion)
                comparison_groups[group_key].extend([df.loc[i].to_dict(), df.loc[j].to_dict()])

    # 3. Final Synthesis (Simulated LLM response for demonstration)
    findings = []
    for i, (group_id, chunks) in enumerate(comparison_groups.items()):
        sources = list(set(c['source_url'] for c in chunks))
        # The LLM would synthesize this summary in the full agent loop
        summary_text = (
            f"Consensus found in {len(sources)} sources regarding '{query_topic}'. "
            f"The core agreement is that multiple independent sources (e.g., {sources[0]}) "
            "are reinforcing the key claim, increasing our confidence."
        )
        findings.append({
            "group_id": f"Group_{i+1}",
            "comparison_type": "Consensus",
            "summary": summary_text,
            "sources": sources
        })
        
    return findings


@mcp.tool
def trend_analysis_tool(knowledge_base: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    """
    Identifies and analyzes temporal patterns (trends) in a specified quantitative metric 
    across the document set.

    Args:
        knowledge_base: The list of enriched document chunks.
        metric_name: The name of the metric to track (e.g., 'cost_per_kwh').

    Returns:
        A dictionary describing the chronological trend and calculated rate of change.
    """
    # 1. Filter and structure data for analysis
    data_points = []
    for chunk in knowledge_base:
        if chunk.get('date') and chunk.get('metrics_extracted') and metric_name in chunk['metrics_extracted']:
            try:
                data_points.append({
                    'date': datetime.strptime(chunk['date'], '%Y-%m-%d'),
                    'value': chunk['metrics_extracted'][metric_name]
                })
            except (ValueError, TypeError):
                continue
                
    if len(data_points) < 2:
        return {"status": "Skipped", "message": "Insufficient data points with date and metric for trend analysis."}

    df = pd.DataFrame(data_points).sort_values('date')
    
    # 2. Calculate trend (Simple Linear Rate of Change)
    start_value = df['value'].iloc[0]
    end_value = df['value'].iloc[-1]
    time_span = (df['date'].iloc[-1] - df['date'].iloc[0]).days
    
    if time_span == 0:
        return {"status": "Skipped", "message": "Time span is too short for meaningful trend analysis."}

    rate_of_change = (end_value - start_value) / time_span * 365 # Annualized rate
    
    trend_type = "Increasing" if rate_of_change > 0 else "Decreasing" if rate_of_change < 0 else "Stable"

    return {
        "analysis_type": "Trend Analysis",
        "metric": metric_name,
        "trend": trend_type,
        "start_date": str(df['date'].iloc[0].date()),
        "end_date": str(df['date'].iloc[-1].date()),
        "total_change": f"{end_value - start_value:.2f}",
        "annualized_rate_of_change": f"{rate_of_change:.2f} per year"
    }

@mcp.tool
def causal_reasoning_tool(knowledge_base: List[Dict[str, Any]], relationship_focus: str) -> Dict[str, Any]:
    """
    Synthesizes facts from the knowledge base to infer or extract cause-and-effect 
    relationships relevant to the focus topic.

    Args:
        knowledge_base: The list of enriched document chunks.
        relationship_focus: The specific relationship to analyze (e.g., 'Nickel mining output on battery costs').

    Returns:
        A structured inference of the cause and effect, ready for Agent 4 validation.
    """
    # 1. Concatenate relevant, high-relevance content for the LLM
    prompt_context = ""
    for chunk in knowledge_base:
        if chunk['relevance_score'] >= 0.8:
            prompt_context += f"- Source {chunk['chunk_id']} ({chunk['source_url']}): {chunk['text_content']}\n"

    if not prompt_context:
        return {"status": "Skipped", "message": "No high-relevance content found for causal analysis."}
        
    # 2. Simulate LLM Call
    # The agent would execute an LLM call with a prompt like:
    llm_prompt = (
        f"ANALYZE the following facts and state the most plausible cause-and-effect relationship "
        f"based ONLY on the provided context, focusing on the relationship: '{relationship_focus}'.\n\nCONTEXT:\n{prompt_context}"
    )

    # Simulated LLM Inference:
    inferred_cause = "Increased nickel mining capacity in late 2022 (Source C003)."
    inferred_effect = "The subsequent drop in the average cost of the Li-ion battery pack (Source C001)."

    return {
        "analysis_type": "Causal Reasoning",
        "focus": relationship_focus,
        "inference": "Plausible Causal Link",
        "cause": inferred_cause,
        "effect": inferred_effect,
        "llm_prompt_used": llm_prompt[:100] + "..." # Truncate for output
    }

@mcp.tool
def statistical_analysis_tool(knowledge_base: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    """
    Performs core descriptive statistics (mean, min, max, variance) on a 
    specified numerical metric across all chunks.

    Args:
        knowledge_base: The list of enriched document chunks.
        metric_name: The name of the metric to analyze (e.g., 'investment').

    Returns:
        A dictionary containing descriptive statistical measures.
    """
    # 1. Collect all numerical values for the target metric
    values = []
    for chunk in knowledge_base:
        if chunk.get('metrics_extracted') and metric_name in chunk['metrics_extracted']:
            value = chunk['metrics_extracted'][metric_name]
            if isinstance(value, (int, float)):
                values.append(value)
                
    if not values:
        return {"status": "Skipped", "message": f"No numerical data found for metric '{metric_name}'."}

    data = np.array(values)
    
    # 2. Calculate descriptive statistics
    return {
        "analysis_type": "Statistical Analysis",
        "metric": metric_name,
        "count": len(data),
        "mean": f"{np.mean(data):.2f}",
        "median": f"{np.median(data):.2f}",
        "standard_deviation": f"{np.std(data):.2f}",
        "minimum": f"{np.min(data):.2f}",
        "maximum": f"{np.max(data):.2f}"
    }
