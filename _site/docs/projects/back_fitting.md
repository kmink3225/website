---
title: "Back Fitting Algorithm"
format: hugo
---



<script  src="../../site_libs/quarto-diagram/mermaid.min.js"></script>
<script  src="../../site_libs/quarto-diagram/mermaid-init.js"></script>
<link  href="../../site_libs/quarto-diagram/mermaid.css" rel="stylesheet" />

## Architecture

<div>

<p>
<pre class="mermaid" data-tooltip-selector="#mermaid-tooltip-1">flowchart LR
    subgraph helper.R
        direction LR
        apply_manual4
        apply_manual3
        gam__single
        apply_3Ct2
        .filtered
        apply_manual2
        apply_manual
        apply_joint3
        subgraph main_run
            direction LR
            subgraph optFG 
                direction LR
                f(parVec)
                .helper
            end
            constrOptim
            gen_uici1
            gen_uici
            get_init
            get_init2
            apply_3Ct
            inverse_L5
            coefs2Ct
            ct2pn
            coefs2pn
            call_pn
            loglike
        end
        subgraph after_fit
            direction LR
            disaggregate
            fct
            sigmoid
            loglike_
            nloglike
            loglike_joint2
            loglike_joint3
            LR_sigmoid
            init_sigmoid4
            summary_sigmoid4
        end
    end
    subgraph visualize.R
        direction LR
        draw_raw
        
        subgraph draw_single
            direction LR
            fct
            subgraph optFG 
                
            end
            
        end
        subgraph draw_3Ct
            direction LR
            disaggregate
           
        end
    end
helper.R --&gt; visualize.R
</pre>

<div id="mermaid-tooltip-1" class="mermaidTooltip">

</div>

</p>

</div>
