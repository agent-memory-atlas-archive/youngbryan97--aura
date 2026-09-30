"""Program-first language variations with independently checked semantic contrasts."""

import hashlib
import random
import string
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import replace
from typing import Any

from core.learning.procedure_induction import Instruction
from core.learning.semantic_graph_counterexamples import (
    ProgramObservationCache,
    compare_program_meanings,
    counterfactual_inputs,
)
from core.learning.semantic_program_campaign import _sha
from core.learning.semantic_program_corpus import (
    SemanticInstructionAnnotation,
    _AnnotatedText,
    _append_natural_binary_operation,
)
from core.learning.semantic_program_floor import (
    compile_source_independent_program_to_floor,
    execute_semantic_floor_program,
    semantic_primitive_type_signature,
    semantic_program_structural_key,
)

_BINARY_LANGUAGE = ('add', 'sub', 'mul', 'idiv', 'at', 'count_of')
_RENDER_STYLES = ('obtain', 'record', 'define', 'name_after')


def _append_bound_expression(text: _AnnotatedText, *, operation: str, ordinal: int,
                             left: str, right: str, labels: tuple[str, str]) -> None:
    """Use a first-operand-first expression with the same typed role spans."""
    label = f'natural:operation:{ordinal}'
    if operation == 'idiv':
        text.append('the whole-number quotient of ', label=label)
    elif operation == 'at':
        text.append('the item of ', label=label)
    elif operation == 'count_of':
        text.append('the count in ', label=label)
    text.append(left, label=labels[0])
    if operation in ('add', 'sub', 'mul'):
        text.append(' ')
        text.append({'add': 'plus', 'sub': 'minus', 'mul': 'times'}[operation],
                    label=label)
        text.append(' ')
    elif operation == 'idiv':
        text.append(' divided by ')
    elif operation == 'at':
        text.append(' at index ')
    elif operation == 'count_of':
        text.append(' of ')
    else:
        raise ValueError('counterfactual expression operation is unsupported')
    text.append(right, label=labels[1])


def render_bound_program(
    source: Any,
    *,
    names: Any,
    clause_order: list[Any],
    program: Any=None,
    lineage_version: int=1,
    render_style: str='obtain',
) -> Any:
    """Render named dependencies independently of their textual clause order."""
    program = source.program if program is None else program
    count = program.n_inputs
    if (source.split != 'train' or count != len(source.inputs)
            or semantic_program_structural_key(program) is None
            or len(names) != count + len(program.instructions)
            or len(set(names)) != len(names)
            or any(not isinstance(name, str) or not name.isascii() or not name.isalpha() for name in names)
            or any(type(index) is not int for index in clause_order)
            or sorted(clause_order) != list(range(len(program.instructions)))
            or any(ins.op not in _BINARY_LANGUAGE for ins in program.instructions)
            or lineage_version not in (1, 2)
            or render_style not in _RENDER_STYLES):
        raise ValueError('counterfactual source or rendering contract is unsupported')
    types = ['integer_sequence' if isinstance(value, tuple) else 'integer' for value in source.inputs]
    for ins in program.instructions:
        signature = semantic_primitive_type_signature(ins.op)
        if tuple(types[index] for index in ins.args) != signature[0]:
            raise ValueError('counterfactual program violates the floor type contract')
        types.append(signature[1])
    try:
        compiled = compile_source_independent_program_to_floor(program, source.inputs,
            provenance_receipt_sha256=_sha({'source': source.example_id, 'program': program.sha()}))
        execution = execute_semantic_floor_program(compiled)
    except (ValueError, TypeError, RuntimeError, ArithmeticError) as exc:
        raise ValueError(f'counterfactual program cannot execute its printed inputs: {exc}') from exc
    if execution.result != program.run(source.inputs):
        raise ValueError('counterfactual floor and primitive execution disagree')
    text = _AnnotatedText()
    text.append('Use these recorded values: ')
    for index, value in enumerate(source.inputs):
        if index:
            text.append('; ')
        text.append(names[index], label=f'definition:{index}')
        text.append(' = ')
        literal = '[' + ', '.join(map(str, value)) + ']' if isinstance(value, tuple) else str(value)
        text.append(literal, label=f'input:{index}')
    text.append('. ')
    annotations = {}
    for ordinal in clause_order:
        ins = program.instructions[ordinal]
        if render_style == 'name_after':
            text.append('Let ')
        else:
            text.append({
                'obtain': 'To obtain ',
                'record': 'Record ',
                'define': 'Define ',
            }[render_style])
            text.append(names[count + ordinal], label=f'definition:{count + ordinal}')
            text.append({
                'obtain': ', ',
                'record': ' as ',
                'define': ' as the result when you ',
            }[render_style])
        left, right = (names[index] for index in ins.args)
        labels = (f'argument:{ordinal}:0', f'argument:{ordinal}:1')
        if render_style in ('record', 'name_after'):
            _append_bound_expression(text, operation=ins.op, ordinal=ordinal,
                                     left=left, right=right, labels=labels)
        elif ins.op in ('at', 'count_of'):
            text.append('select the item at' if ins.op == 'at' else 'count', label=f'natural:operation:{ordinal}')
            text.append(' ' if ins.op == 'at' else ' how often ')
            text.append(right, label=labels[1])
            text.append(' in ' if ins.op == 'at' else ' occurs in ')
            text.append(left, label=labels[0])
        else:
            _append_natural_binary_operation(text, op=ins.op, ordinal=ordinal,
                left_text=left, left_label=labels[0], right_text=right, right_label=labels[1])
        if render_style == 'name_after':
            text.append(' be named ')
            text.append(names[count + ordinal], label=f'definition:{count + ordinal}')
        text.append('. ')
        annotations[ordinal] = SemanticInstructionAnnotation(ins, text.span(f'natural:operation:{ordinal}'),
            tuple(text.span(label) for label in labels), tuple(sorted({arg - count for arg in ins.args if arg >= count})))
    text.append('Return ' + names[-1] + '.')
    identity = _sha({'source': source.example_id, 'text': text.text, 'program': program.sha()})
    return replace(source, example_id=identity,
        construction_id=(f'counterfactual-bound-v2:{source.construction_id}'
                         if lineage_version == 2 else 'counterfactual-bound-v1'),
        topology_id=_sha(semantic_program_structural_key(program)), source_text=text.text,
        input_spans=tuple(text.span(f'input:{index}') for index in range(count)),
        instructions=tuple(annotations[index] for index in range(len(annotations))),
        report_value=count + len(annotations) - 1,
        contrast_id=(hashlib.sha256(source.source_text.encode('utf-8')).hexdigest()
                     if lineage_version == 2 else source.example_id),
        register_definition_spans=tuple(text.span(f'definition:{index}') for index in range(len(names))))


def _names(rng: Any, count: Any) -> tuple[Any, ...]:
    names = []
    while len(names) < count:
        value = ''.join(rng.choice(string.ascii_lowercase) for _ in range(8))
        if value not in names:
            names.append(value)
    return tuple(names)


def equivalent_recompositions(program: Any) -> Iterator[Any]:
    """Rotate single-use associative subtrees without changing public inputs."""
    for child, instruction in enumerate(program.instructions):
        if instruction.op not in ('add', 'mul'):
            continue
        register = program.n_inputs + child
        uses = [(index, slot) for index, ins in enumerate(program.instructions)
                for slot, argument in enumerate(ins.args) if argument == register]
        if len(uses) != 1:
            continue
        parent, slot = uses[0]
        parent_instruction = program.instructions[parent]
        if parent_instruction.op != instruction.op:
            continue
        other = parent_instruction.args[1 - slot]
        if other >= register:
            continue
        left, right = instruction.args
        changed = list(program.instructions)
        changed[child] = Instruction(instruction.op, (right, other))
        changed[parent] = Instruction(instruction.op, (left, register))
        candidate = replace(program, instructions=tuple(changed))
        if (candidate != program
                and compare_program_meanings(program, candidate, ())['status'] == 'equivalent'):
            yield candidate


def _source_relation_records(examples: tuple[Any, ...]) -> dict[tuple, list[tuple]]:
    grouped: dict[tuple, list[tuple]] = defaultdict(list)
    identities = set()
    for item in examples:
        if item.split != 'train':
            raise ValueError('relation controls require source training only')
        ir = getattr(item, 'ir', None)
        source_id = item.example_id if ir is None else ir.source_text_sha256
        program = item.program if ir is None else ir.to_program()
        inputs = item.inputs if ir is None else item.public_inputs
        source_hash = (hashlib.sha256(item.source_text.encode('utf-8')).hexdigest()
                       if ir is None else ir.source_text_sha256)
        if source_id in identities:
            raise ValueError('relation controls repeat a source identity')
        identities.add(source_id)
        relation = semantic_program_structural_key(program)
        if relation is None:
            raise ValueError('relation controls require connected typed programs')
        record = (source_id, item.construction_id, item.contrast_id or source_id,
                  program, inputs, source_hash)
        grouped[relation].append(record)
    return grouped


def cross_construction_relation_partners(examples: tuple[Any, ...]) -> dict[str, str]:
    """Give each fit source an independent same-relation construction partner."""
    partners = {}
    for records in _source_relation_records(examples).values():
        ordered = sorted(records)
        for source in ordered:
            other = next((candidate for candidate in ordered
                          if candidate[1] != source[1] and candidate[2] != source[2]), None)
            if other is not None:
                partners[source[0]] = other[0]
    return partners


def cross_construction_relation_triplets(examples: tuple[Any, ...]) -> dict[str, tuple[str, str]]:
    """Match a relation across forms and witness a rival within a form.

    All three sources come from the caller's training split. The same-form
    negative stops a representation from solving the task by construction ID.
    A different structural key alone does not prove different meaning.
    """
    grouped = _source_relation_records(examples)
    all_records = sorted((record, relation) for relation, records in grouped.items()
                         for record in records)
    observations = ProgramObservationCache(capacity=512)
    triplets = {}
    for source, relation in all_records:
        positive = next((other for other in sorted(grouped[relation])
                         if other[1] != source[1] and other[2] != source[2]), None)
        if positive is None:
            continue
        negative = None
        for other, other_relation in all_records:
            if (other_relation == relation or other[1] != source[1]
                    or other[2] == source[2] or other[3].n_inputs != source[3].n_inputs
                    or tuple(type(value) for value in other[4])
                    != tuple(type(value) for value in source[4])):
                continue
            probes = counterfactual_inputs(source[4], count=8)
            comparison = compare_program_meanings(
                source[3], other[3], probes, observation_cache=observations)
            if comparison['status'] == 'different' and comparison.get('witness') is not None:
                negative = other
                break
        if positive is not None and negative is not None:
            triplets[source[0]] = (positive[0], negative[0])
    return triplets


def cross_construction_relation_controls(examples: tuple[Any, ...]) -> dict[str, Any]:
    """Pair shared computation across independent source forms with witnessed rivals.

    The returned pairs are source supervision, not evidence that a decoder
    recognizes the relation in a new utterance. Contrast lineage never crosses
    a pair, and a merely type-compatible but unproved rival is not a negative.
    """
    from itertools import combinations

    from core.learning.semantic_candidate_contrasts import source_program_factor_contrasts

    grouped = _source_relation_records(examples)
    identities = {record[0] for records in grouped.values() for record in records}
    pairs = []
    for relation, records in sorted(grouped.items(), key=lambda row: _sha(row[0])):
        by_construction = {}
        for record in sorted(records):
            by_construction.setdefault(record[1], record)
        for left, right in combinations((by_construction[key] for key in sorted(by_construction)), 2):
            if left[2] == right[2]:
                continue
            candidates = source_program_factor_contrasts(
                left[3], left[4], source_sha256=left[5])
            if len(candidates) < 2:
                continue
            role_rivals = [candidate for candidate in candidates[1:] if all(
                proposed.op == original.op for proposed, original in zip(
                    candidate.instructions, left[3].instructions, strict=True))]
            rival = role_rivals[0] if role_rivals else candidates[1]
            negative_kind = 'role_or_dependency_flip' if role_rivals else 'operation_change'
            comparison = compare_program_meanings(left[3], rival,
                                                  counterfactual_inputs(left[4]))
            if comparison['status'] != 'different' or comparison.get('witness') is None:
                raise ValueError('relation rival lacks a changed-meaning witness')
            pairs.append({'left': left[0], 'right': right[0],
                          'left_construction': left[1],
                          'right_construction': right[1],
                          'relation_sha256': _sha(relation),
                          'positive_program_sha256': left[3].sha(),
                          'negative_program_sha256': rival.sha(),
                          'negative_kind': negative_kind,
                          'negative_witness': comparison['witness']})
    body = {'schema': 'aura.semantic_cross_construction_relation_controls.v1',
            'source_examples': len(identities), 'relations': len(grouped),
            'cross_construction_pairs': len(pairs), 'pairs': pairs,
            'validation_or_test_examples_used': 0, 'serving_authority': False}
    return {**body, 'receipt_sha256': _sha(body)}


def augment_source_programs(
    examples: tuple[Any, ...],
    *,
    seed: int=0,
    variations: int=2,
    forbidden_constructions: tuple[Any, ...]=(),
    lineage_version: int=1,
    mutation_policy: str='first',
    render_styles: tuple[str, ...]=('obtain',),
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Rename/reorder source programs and retain only witnessed meaning changes."""
    if (type(seed) is not int or type(variations) is not int or variations < 1
            or lineage_version not in (1, 2)
            or mutation_policy not in ('first', 'all_witnessed')
            or not render_styles or len(set(render_styles)) != len(render_styles)
            or any(style not in _RENDER_STYLES for style in render_styles)):
        raise ValueError('counterfactual generation settings are invalid')
    examples = tuple(examples)
    sources = tuple(item for item in examples if item.split == 'train')
    if len({item.example_id for item in sources}) != len(sources):
        raise ValueError('counterfactual sources repeat an identity')
    if any(item.construction_id in forbidden_constructions for item in sources):
        raise ValueError('counterfactual source construction is reserved for evaluation')
    rows, records = [], []
    for source in sources:
        rng = random.Random(_sha({'seed': seed, 'source': source.example_id}))
        names = _names(rng, len(source.inputs) + len(source.instructions))
        for variant in range(variations):
            order = list(range(len(source.instructions)))
            if variant:
                if variant == 1:
                    order.reverse()
                else:
                    rng.shuffle(order)
                names = _names(rng, len(names))
            try:
                rendered = render_bound_program(source, names=names, clause_order=order,
                                                lineage_version=lineage_version,
                                                render_style=render_styles[variant % len(render_styles)])
            except ValueError as exc:
                records.append({'source': source.example_id, 'status': 'unsupported', 'reason': str(exc)})
                break
            proof = compare_program_meanings(source.program, rendered.program, ())
            assert proof['status'] == 'equivalent'
            rows.append(rendered)
            records.append({'source': source.example_id, 'example': rendered.example_id,
                            'status': 'equivalent', 'comparison': proof})
        else:
            for program in equivalent_recompositions(source.program):
                rendered = render_bound_program(source, names=names, clause_order=order,
                                                program=program, lineage_version=lineage_version,
                                                render_style=render_styles[0])
                rows.append(rendered)
                records.append({'source': source.example_id, 'example': rendered.example_id,
                    'status': 'equivalent', 'transformation': 'associative_recomposition',
                    'comparison': compare_program_meanings(source.program, program, ())})
            # Mutations operate on typed SSA edges and primitives, never on answer labels.
            mutations = []
            for index, annotation in enumerate(source.instructions):
                ins = annotation.instruction
                for operation in _BINARY_LANGUAGE:
                    if operation != ins.op and semantic_primitive_type_signature(operation) == semantic_primitive_type_signature(ins.op):
                        mutations.append((index, Instruction(operation, ins.args)))
                if len(set(ins.args)) == 2:
                    mutations.append((index, Instruction(ins.op, tuple(reversed(ins.args)))))
            rng.shuffle(mutations)
            witnessed = 0
            for mutation_ordinal, (index, instruction) in enumerate(mutations):
                program = replace(source.program, instructions=tuple(
                    instruction if ordinal == index else ann.instruction for ordinal, ann in enumerate(source.instructions)))
                try:
                    rendered = render_bound_program(source, names=names, clause_order=order,
                                                    program=program, lineage_version=lineage_version,
                                                    render_style=render_styles[mutation_ordinal % len(render_styles)])
                except ValueError:
                    continue
                comparison = compare_program_meanings(source.program, program, counterfactual_inputs(source.inputs))
                if comparison['status'] != 'different' or comparison.get('witness') is None:
                    continue
                rows.append(rendered)
                record = {'source': source.example_id, 'example': rendered.example_id,
                          'status': 'different', 'comparison': comparison}
                if mutation_policy == 'all_witnessed':
                    record.update(mutation_step=index,
                                  mutation_kind=('operation' if instruction.op !=
                                                 source.instructions[index].instruction.op else 'role'))
                records.append(record)
                witnessed += 1
                if mutation_policy == 'first':
                    break
            if not witnessed:
                records.append({'source': source.example_id, 'status': 'no_witnessed_mutation'})
    body = {'schema': 'aura.semantic_counterfactual_corpus.v1', 'seed': seed, 'variations': variations,
        'source_training_ids': sorted(item.example_id for item in sources),
        'ignored_nontraining_examples': sum(item.split != 'train' for item in examples),
        'forbidden_constructions': sorted(forbidden_constructions), 'records': records,
        'generated_examples': len(rows), 'test_examples_used': 0, 'serving_authority': False}
    if lineage_version == 2:
        body['lineage_version'] = 2
    if mutation_policy != 'first':
        body['mutation_policy'] = mutation_policy
    if render_styles != ('obtain',):
        body['render_styles'] = list(render_styles)
    return tuple(rows), {**body, 'receipt_sha256': _sha(body)}


def build_semantic_counterfactual_source_corpus(
    *,
    seed: int=0,
    examples_per_schema_domain: int=1,
    lineage_version: int=1,
) -> Any:
    """Augment existing natural source training, never its validation/test domains."""
    from core.learning.semantic_program_corpus_natural import (
        build_semantic_program_natural_source_corpus,
    )

    sources = build_semantic_program_natural_source_corpus(
        seed=seed, examples_per_schema_domain=examples_per_schema_domain)
    rows, receipt = augment_source_programs(sources, seed=seed,
                                            lineage_version=lineage_version)
    if any(row['status'] == 'unsupported' for row in receipt['records']):
        raise ValueError('declared counterfactual source corpus has unsupported programs')
    return rows


def build_semantic_counterfactual_fork_join_corpus(
    *, seed: int=0, examples_per_operation_triple: int=1,
) -> Any:
    """Cross fit wordings/topologies before making witnessed step contrasts."""
    from core.learning.semantic_program_corpus import (
        build_semantic_program_fork_join_corpus,
        build_semantic_program_fork_join_factorial_corpus,
    )

    fit_topologies = {item.topology_id for item in build_semantic_program_fork_join_corpus(
        seed=seed, examples_per_operation_triple=1) if item.split == 'train'}
    sources = tuple(replace(item, contrast_id=hashlib.sha256(
        item.source_text.encode('utf-8')).hexdigest())
        for item in build_semantic_program_fork_join_factorial_corpus(
        seed=seed, examples_per_cell=examples_per_operation_triple)
        if item.split == 'train' and item.topology_id in fit_topologies)
    rows, receipt = augment_source_programs(
        sources, seed=seed, variations=len(_RENDER_STYLES), lineage_version=2,
        mutation_policy='all_witnessed', render_styles=_RENDER_STYLES)
    if any(row['status'] == 'unsupported' for row in receipt['records']):
        raise ValueError('fork/join counterfactual source corpus has unsupported programs')
    return (*sources, *rows)


def build_semantic_counterfactual_fork_join_stop_corpus(
    *, seed: int=0, examples_per_operation_triple: int=1,
) -> Any:
    """Pair complete fit graphs with a witnessed continuation at their stop choice.

    The base set is the deterministic first row of each v1 factorial cell. The
    extra step reuses an existing input, so both graphs consume the same public
    inputs and share every teacher decision before finish versus continue.
    """
    from core.learning.procedure_induction import Instruction

    originals = build_semantic_counterfactual_fork_join_corpus(
        seed=seed, examples_per_operation_triple=examples_per_operation_triple)
    cells = {}
    for item in originals:
        key = (item.construction_id, item.topology_id,
               tuple(row.instruction.op for row in item.instructions))
        if key not in cells or item.example_id < cells[key].example_id:
            cells[key] = item
    base = tuple(cells[key] for key in sorted(cells))
    extended = []
    observations = ProgramObservationCache(capacity=512)
    for item in base:
        program = item.program
        if (program.n_inputs != 4 or program.depth != 3
                or any(type(value) is not int for value in item.inputs)):
            raise ValueError('stop contrast requires a three-step integer fork/join graph')
        continuation = replace(program, instructions=(*program.instructions,
            Instruction('add', (program.n_inputs + program.depth - 1, 0))))
        comparison = compare_program_meanings(program, continuation,
            counterfactual_inputs(item.inputs), observation_cache=observations)
        if comparison['status'] != 'different' or comparison.get('witness') is None:
            raise ValueError('stop contrast has no witnessed output change')
        rng = random.Random(f'{seed}|{item.example_id}|stop')
        rendered = render_bound_program(item, program=continuation,
            names=_names(rng, program.n_inputs + continuation.depth),
            clause_order=list(range(continuation.depth)), lineage_version=2,
            render_style='name_after')
        extended.append(replace(rendered, contrast_id=item.contrast_id))
    if len({item.example_id for item in (*base, *extended)}) != 2 * len(base):
        raise ValueError('stop contrast source identities are not unique')
    return (*base, *extended)
