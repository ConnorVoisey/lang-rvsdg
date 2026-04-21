use llvm_ir::Name;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BasicBlockId(pub u32);

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlockMapper {
    map: FxHashMap<Name, BasicBlockId>,
    /// the last block is always the exit block
    pub(crate) blocks: Vec<BasicBlockInOuts>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BasicBlockInOuts {
    pub(crate) inputs: SmallVec<[BasicBlockId; 4]>,
    pub(crate) outputs: SmallVec<[BasicBlockId; 4]>,
}

impl BasicBlockMapper {
    pub fn new(capacity: usize) -> Self {
        Self {
            map: FxHashMap::default(),
            blocks: Vec::with_capacity(capacity),
        }
    }
    #[inline]
    pub fn get(&self, name: &Name) -> Option<&BasicBlockId> {
        self.map.get(name)
    }

    #[inline]
    pub fn get_expect(&self, name: &Name) -> &BasicBlockId {
        self.get(name)
            .expect("expected block with name: {name} to already be interned")
    }

    #[inline]
    pub fn get_exit(&self) -> Option<&BasicBlockId> {
        self.map.iter().last().map(|(_, v)| v)
    }

    #[inline]
    pub fn exit_name(&self) -> Name {
        llvm_ir::Name::Number(u32::MAX as usize)
    }
    #[inline]
    pub fn get_exit_expect(&self) -> &BasicBlockId {
        let val = self
            .get_exit()
            .expect("exit block should have been interned");

        // check that the last block is actually the exit block
        debug_assert!(match self.get(&self.exit_name()) {
            Some(v) => v == val,
            None => false,
        });

        val
    }

    #[inline]
    pub fn intern(&mut self, name: &Name) -> BasicBlockId {
        match self.get(name) {
            Some(id) => *id,
            None => {
                let id = BasicBlockId(self.blocks.len() as u32);
                self.map.insert(name.clone(), id);
                self.blocks.push(BasicBlockInOuts {
                    inputs: SmallVec::new(),
                    outputs: SmallVec::new(),
                });
                id
            }
        }
    }

    pub fn add_connection(&mut self, from: BasicBlockId, to: BasicBlockId) {
        self.blocks[from.0 as usize].outputs.push(to);
        self.blocks[to.0 as usize].inputs.push(from);
    }

    pub fn inputs(&self, id: BasicBlockId) -> &[BasicBlockId] {
        &self.blocks[id.0 as usize].inputs
    }

    pub fn outputs(&self, id: BasicBlockId) -> &[BasicBlockId] {
        &self.blocks[id.0 as usize].outputs
    }
}
