/* globals $ */
export default class MuskiPerformanceRnnUi {
  public $element: JQuery<HTMLElement>;
  private $noteDensityInput: JQuery<HTMLElement>;
  private $gainInput: JQuery<HTMLElement>;
  private $status: JQuery<HTMLElement>;

  constructor(options : object = {}) {
    this.$element = $('<div></div>')
      .addClass('muski-performance-rnn-ui');

    this.$noteDensityInput = $('<input />')
      .attr('type', 'range')
      .addClass(['form-range'])
      .attr({
        id: 'note-density',
        min: 0,
        max: 6,
        value: 2,
      });

    this.$gainInput = $('<input />')
      .attr('type', 'range')
      .addClass(['form-range'])
      .attr({
        id: 'gain',
        min: 0,
        max: 200,
        value: 100,
      });

    this.$status = $('<div></div>')
      .attr('id', 'resettingText')
      .addClass(['text-center', 'mb-3'])
      .css('opacity', 0)
      .append(
        $('<div></div>')
          .addClass(['badge', 'text-bg-warning'])
          .text('Resetting...')
      );

    $('<div></div>')
      .addClass(['row', 'justify-content-center'])
      .append($('<div></div>')
        .addClass(['col-lg-6', 'col-md-8'])
        .append($('<div></div>')
          .addClass(['row', 'mt-4 mb-3'])
          .append($('<div></div>')
            .addClass(['col'])
            .append($('<div></div>')
              .append($('<label></label>')
                .attr('for', 'note-density')
                .addClass(['form-label', 'd-block', 'mb-0'])
                .html('Note Density (<span id="note-density-display"></span>)'))
            )
            .append(this.$noteDensityInput)
            .append($('<label></label>')
              .attr('for', 'gain')
              .addClass(['form-label', 'd-block', 'mb-0'])
              .html('Gain (<span id="gain-display"></span>%)'))
            .append(this.$gainInput)
          )
          .append($('<div></div>')
            .addClass(['col'])
            .append($('<button></button>')
              .attr({
                type: 'button',
                id: 'start-pause-button',
                disabled: true,
              })
              .addClass(['btn', 'btn-primary', 'btn-lg', 'disabled', 'me-2'])
              .text('Play'))
            .append($('<button></button>')
              .attr({
                type: 'button',
                id: 'reset-rnn',
              })
              .addClass(['btn', 'btn-warning'])
              .text('Reset RNN'))
          )
        )
        .append($('<div></div>')
          .addClass(['col'])
          .append(
            this.$status
          )
        ))
      .appendTo(this.$element);
  }
}
