use color_eyre::eyre::eyre;

use crate::rvsdg::RVSDGMod;

pub mod dead_node_elimination;

#[derive(Debug)]
pub enum OptPass {
    DeadNodeElimination,
}

impl OptPass {
    fn run_pass(&self, rvsdg_mod: &mut RVSDGMod) -> color_eyre::Result<()> {
        match self {
            OptPass::DeadNodeElimination => rvsdg_mod.opt_dead_node_elimination(),
        }
    }
}

impl RVSDGMod {
    /// Run `passes` in order. With `verify_all`, the whole graph is
    /// verified before the first pass (naming construction as the
    /// culprit rather than the pass that trips over it) and after every
    /// pass, so a broken invariant is attributed to the pass that broke
    /// it instead of surfacing as a miscompile downstream.
    pub fn optimise(&mut self, passes: Vec<OptPass>, verify_all: bool) -> color_eyre::Result<()> {
        if verify_all {
            self.verify_stage("before any passes")?;
        }
        for pass in passes {
            pass.run_pass(self)?;
            if verify_all {
                self.verify_stage(&format!("after {pass:?}"))?;
            }
        }
        Ok(())
    }

    pub fn optimise_default(&mut self, verify_all: bool) -> color_eyre::Result<()> {
        let passes = vec![OptPass::DeadNodeElimination];
        self.optimise(passes, verify_all)
    }

    fn verify_stage(&self, stage: &str) -> color_eyre::Result<()> {
        let errs = self.verify();
        if errs.is_empty() {
            return Ok(());
        }
        let listed: Vec<String> = errs.iter().take(10).map(|e| format!("  {e}")).collect();
        Err(eyre!(
            "verification failed {stage}: {} errors\n{}{}",
            errs.len(),
            listed.join("\n"),
            if errs.len() > listed.len() {
                "\n  ..."
            } else {
                ""
            },
        ))
    }
}
